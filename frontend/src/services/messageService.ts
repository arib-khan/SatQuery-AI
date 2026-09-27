import {
  collection,
  doc,
  setDoc,
  updateDoc,
  serverTimestamp,
  query,
  orderBy,
  limit as fsLimit,
  startAfter,
  getDocs,
  onSnapshot,
  type QueryDocumentSnapshot,
  type Unsubscribe,
} from 'firebase/firestore';
import { getDb } from '@/lib/firebase/client';
import type { ChatMessage, ImageRef, MessageMetadata, MessageRole, MessageStatus } from '@/types/chat';

const messagesCol = async (uid: string, conversationId: string) =>
  collection(await getDb(), 'users', uid, 'conversations', conversationId, 'messages');
const messageDoc = async (uid: string, conversationId: string, messageId: string) =>
  doc(await getDb(), 'users', uid, 'conversations', conversationId, 'messages', messageId);

const PAGE_SIZE = 30;

function fromSnap(d: QueryDocumentSnapshot, conversationId: string): ChatMessage {
  const data = d.data();
  return {
    messageId: d.id,
    conversationId,
    role: data.role,
    content: data.content ?? '',
    images: data.images ?? [],
    createdAt: data.createdAt ?? null,
    status: data.status ?? 'completed',
    metadata: data.metadata ?? undefined,
  };
}

/**
 * Writes a message with a caller-supplied deterministic id (crypto.randomUUID()).
 * Using setDoc with a fixed id instead of addDoc means retrying the same
 * logical send (e.g. after a flaky network write) overwrites the same
 * document instead of creating a duplicate message.
 */
export async function saveMessage(
  uid: string,
  conversationId: string,
  messageId: string,
  params: { role: MessageRole; content: string; images: ImageRef[]; status: MessageStatus; metadata?: MessageMetadata }
): Promise<void> {
  await setDoc(await messageDoc(uid, conversationId, messageId), {
    role: params.role,
    content: params.content,
    images: params.images,
    status: params.status,
    metadata: params.metadata ?? null,
    createdAt: serverTimestamp(),
  });
}

export async function updateMessageStatus(
  uid: string,
  conversationId: string,
  messageId: string,
  updates: Partial<{ content: string; images: ImageRef[]; status: MessageStatus; metadata: MessageMetadata }>
): Promise<void> {
  await updateDoc(await messageDoc(uid, conversationId, messageId), updates);
}

/** Most recent page of messages, oldest-first for display. Used for the initial conversation load. */
export async function fetchLatestMessages(uid: string, conversationId: string): Promise<{ messages: ChatMessage[]; hasMore: boolean }> {
  const q = query(await messagesCol(uid, conversationId), orderBy('createdAt', 'desc'), fsLimit(PAGE_SIZE));
  const snap = await getDocs(q);
  const messages = snap.docs.map((d) => fromSnap(d, conversationId)).reverse();
  return { messages, hasMore: snap.docs.length === PAGE_SIZE };
}

/** Loads an older page (for "load more" / infinite-scroll-up pagination). */
export async function fetchOlderMessages(
  uid: string,
  conversationId: string,
  beforeDoc: QueryDocumentSnapshot
): Promise<{ messages: ChatMessage[]; hasMore: boolean }> {
  const q = query(await messagesCol(uid, conversationId), orderBy('createdAt', 'desc'), startAfter(beforeDoc), fsLimit(PAGE_SIZE));
  const snap = await getDocs(q);
  const messages = snap.docs.map((d) => fromSnap(d, conversationId)).reverse();
  return { messages, hasMore: snap.docs.length === PAGE_SIZE };
}

/**
 * Live subscription for the tail of the conversation (keeps the open chat in
 * sync). Returns an Unsubscribe synchronously — see the matching comment in
 * conversationService.ts's subscribeToConversations for why, and why that's
 * safe even though getDb() itself is async.
 */
export function subscribeToRecentMessages(
  uid: string,
  conversationId: string,
  onChange: (messages: ChatMessage[]) => void,
  onError: (err: Error) => void
): Unsubscribe {
  let unsub: Unsubscribe | null = null;
  let cancelled = false;

  (async () => {
    try {
      const col = await messagesCol(uid, conversationId);
      if (cancelled) return;
      const q = query(col, orderBy('createdAt', 'desc'), fsLimit(PAGE_SIZE));
      unsub = onSnapshot(
        q,
        (snap) => onChange(snap.docs.map((d) => fromSnap(d, conversationId)).reverse()),
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