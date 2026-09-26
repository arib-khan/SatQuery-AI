'use client';

import { useEffect, useState, useCallback } from 'react';
import { collection, query, orderBy, limit, startAfter, getDocs } from 'firebase/firestore';
import { getDb } from '@/lib/firebase/client';
import { useAuth } from '@/contexts/AuthContext';
import { subscribeToRecentMessages } from '@/services/messageService';
import type { ChatMessage } from '@/types/chat';

const PAGE_SIZE = 30;

export function useMessages(conversationId: string | null) {
  const { user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user || !conversationId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clears stale data synchronously when there's no active conversation; no subscription is started in this branch.
      setMessages([]);
      setHasMore(false);
      return;
    }
    setLoading(true);
    setError(null);
    const unsubscribe = subscribeToRecentMessages(
      user.uid,
      conversationId,
      (items) => {
        setMessages(items);
        setLoading(false);
        // Recent-page fetch may have more history below it; only known for
        // certain once a "load more" request comes back short.
        setHasMore(items.length >= PAGE_SIZE);
      },
      (err) => {
        setError(err.message || 'Could not load this conversation.');
        setLoading(false);
      }
    );
    return unsubscribe;
  }, [user, conversationId]);

  const loadOlder = useCallback(async () => {
    if (!user || !conversationId || messages.length === 0 || loadingMore) return;
    setLoadingMore(true);
    try {
      const db = await getDb();
      const col = collection(db, 'users', user.uid, 'conversations', conversationId, 'messages');
      // Anchor pagination on the oldest currently-loaded message's timestamp.
      const oldest = messages[0];
      const q = query(col, orderBy('createdAt', 'desc'), startAfter(oldest.createdAt), limit(PAGE_SIZE));
      const snap = await getDocs(q);
      const older = snap.docs
        .map((d) => ({
          messageId: d.id,
          conversationId,
          role: d.data().role,
          content: d.data().content ?? '',
          images: d.data().images ?? [],
          createdAt: d.data().createdAt ?? null,
          status: d.data().status ?? 'completed',
          metadata: d.data().metadata ?? undefined,
        }))
        .reverse();
      setMessages((prev) => [...older, ...prev]);
      setHasMore(snap.docs.length === PAGE_SIZE);
    } catch (err) {
      setError((err as Error).message || 'Could not load older messages.');
    } finally {
      setLoadingMore(false);
    }
  }, [user, conversationId, messages, loadingMore]);

  return { messages, loading, loadingMore, hasMore, loadOlder, error };
}
