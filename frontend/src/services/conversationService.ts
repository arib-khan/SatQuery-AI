/**
 * All reads/writes here go through the Firebase CLIENT SDK and are scoped to
 * `users/{uid}/conversations/...`. Cross-user access is impossible even if a
 * caller passes the wrong uid, because firestore.rules independently enforces
 * `request.auth.uid == uid` on every document under a user's tree — this
 * service is a convenience layer, not the security boundary.
 */
import {
  collection,
  doc,
  serverTimestamp,
  addDoc,
  updateDoc,
  onSnapshot,
  query,
  orderBy,
  limit as fsLimit,
  getDocs,
  writeBatch,
  type Unsubscribe,
} from 'firebase/firestore';
import { getDb } from '@/lib/firebase/client';
import type { Conversation } from '@/types/chat';

const conversationsCol = async (uid: string) => collection(await getDb(), 'users', uid, 'conversations');
const conversationDoc = async (uid: string, conversationId: string) => doc(await getDb(), 'users', uid, 'conversations', conversationId);
const messagesCol = async (uid: string, conversationId: string) =>
  collection(await getDb(), 'users', uid, 'conversations', conversationId, 'messages');

/** Derives a short, meaningful title from the first user message without calling an LLM. */
export function deriveTitleFromMessage(text: string): string {
  const cleaned = text.trim().replace(/\s+/g, ' ');
  if (!cleaned) return 'New mission';
  const words = cleaned.split(' ');
  const truncated = words.slice(0, 9).join(' ');
  return truncated.length < cleaned.length ? `${truncated}…` : truncated;
}

export async function createConversation(uid: string, firstMessageText: string): Promise<string> {
  const ref = await addDoc(await conversationsCol(uid), {
    title: deriveTitleFromMessage(firstMessageText),
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
    lastMessage: firstMessageText.slice(0, 200),
    messageCount: 0,
  });
  return ref.id;
}

/**
 * Returns an Unsubscribe synchronously (so it plugs directly into a
 * useEffect's cleanup, like the caller expects) even though obtaining the
 * Firestore instance itself is async (see lib/firebase/client.ts's getDb for
 * why db can never be a plain top-level constant). The real listener attaches
 * once getDb() resolves; if the caller unsubscribes before that happens, the
 * `cancelled` flag makes sure the listener is torn down immediately once it
 * does attach instead of leaking.
 */
export function subscribeToConversations(
  uid: string,
  onChange: (conversations: Conversation[]) => void,
  onError: (err: Error) => void
): Unsubscribe {
  let unsub: Unsubscribe | null = null;
  let cancelled = false;

  (async () => {
    try {
      const col = await conversationsCol(uid);
      if (cancelled) return;
      const q = query(col, orderBy('updatedAt', 'desc'), fsLimit(200));
      unsub = onSnapshot(
        q,
        (snap) => {
          const items: Conversation[] = snap.docs.map((d) => ({
            conversationId: d.id,
            title: d.data().title ?? 'Untitled mission',
            createdAt: d.data().createdAt ?? null,
            updatedAt: d.data().updatedAt ?? null,
            lastMessage: d.data().lastMessage ?? '',
            messageCount: d.data().messageCount ?? 0,
          }));
          onChange(items);
        },
        (err) => onError(err as Error)
      );
      if (cancelled) unsub();
    } catch (err) {
      if (!cancelled) onError(err as Error);
    }
  })();

  return () => {
    cancelled = true;
    unsub?.();
  };
}

export async function renameConversation(uid: string, conversationId: string, title: string): Promise<void> {
  const trimmed = title.trim();
  if (!trimmed) return;
  await updateDoc(await conversationDoc(uid, conversationId), { title: trimmed.slice(0, 120) });
}

export async function touchConversation(uid: string, conversationId: string, lastMessage: string, messageCountDelta: number, currentCount: number): Promise<void> {
  await updateDoc(await conversationDoc(uid, conversationId), {
    updatedAt: serverTimestamp(),
    lastMessage: lastMessage.slice(0, 200),
    messageCount: currentCount + messageCountDelta,
  });
}

/**
 * Deletes a conversation and all of its messages. Firestore doesn't cascade-
 * delete subcollections automatically, so messages are fetched and removed
 * in a batch first. Cloudinary image cleanup for any images referenced by
 * those messages is handled by the caller (see hooks/useConversations.ts),
 * since it requires calling the server-side delete route, not Firestore.
 */
export async function deleteConversationDeep(uid: string, conversationId: string): Promise<void> {
  const database = await getDb();
  const messagesSnap = await getDocs(await messagesCol(uid, conversationId));
  const batch = writeBatch(database);
  messagesSnap.docs.forEach((d) => batch.delete(d.ref));
  batch.delete(await conversationDoc(uid, conversationId));
  await batch.commit();
}

export function groupConversationsByRecency(conversations: Conversation[]) {
  const now = Date.now();
  const DAY = 86_400_000;
  const groups: { label: string; items: Conversation[] }[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 Days', items: [] },
    { label: 'Previous 30 Days', items: [] },
    { label: 'Older', items: [] },
  ];

  for (const c of conversations) {
    const ts = c.updatedAt?.toMillis?.() ?? 0;
    const ageDays = (now - ts) / DAY;
    if (ageDays < 1) groups[0].items.push(c);
    else if (ageDays < 2) groups[1].items.push(c);
    else if (ageDays < 7) groups[2].items.push(c);
    else if (ageDays < 30) groups[3].items.push(c);
    else groups[4].items.push(c);
  }

  return groups.filter((g) => g.items.length > 0);
}