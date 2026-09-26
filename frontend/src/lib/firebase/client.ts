/**
 * Firebase CLIENT SDK initialization.
 *
 * IMPORTANT: this file only ever touches `NEXT_PUBLIC_*` environment variables.
 * Firebase web "config" values (apiKey, authDomain, etc.) are not secrets — they
 * identify the project, and access is actually enforced by Firestore Security
 * Rules (see /firestore.rules) plus Firebase Auth. Never put the Admin SDK
 * service-account key here; that lives only in lib/firebase/admin.ts and is
 * used exclusively in server-only route handlers.
 *
 * This module is a singleton so every component/hook shares one initialized
 * app/auth/firestore instance instead of re-initializing on every render.
 */
import { initializeApp, getApps, getApp, type FirebaseApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, type Auth } from 'firebase/auth';
// Type-only import: erased entirely at compile time, so it has zero runtime
// footprint and does NOT pull the firebase/firestore module into any bundle
// on its own. See getDb() below for why the real module must only ever be
// loaded dynamically, in the browser.
import type { Firestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

function assertConfigured() {
  const missing = Object.entries(firebaseConfig)
    .filter(([, v]) => !v)
    .map(([k]) => k);
  if (missing.length > 0 && typeof window !== 'undefined') {
    // Non-fatal warning only in the browser; lets the build succeed without
    // real credentials while making misconfiguration obvious at runtime.
    console.warn(
      `[firebase] Missing NEXT_PUBLIC_FIREBASE_* env vars: ${missing.join(', ')}. ` +
      'Authentication and chat history will not work until frontend/.env.local is configured.'
    );
  }
}

assertConfigured();

export const firebaseApp: FirebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig);

// getAuth() is guarded to only run in the browser as a defensive measure
// (Auth's platform-detection touches a few browser-only globals); confirmed
// safe to import statically (unlike Firestore below) since it doesn't pull
// in protobufjs.
export const auth: Auth = typeof window !== 'undefined' ? getAuth(firebaseApp) : (null as unknown as Auth);
export const googleProvider = new GoogleAuthProvider();

export function isFirebaseConfigured(): boolean {
  return Boolean(firebaseConfig.apiKey && firebaseConfig.projectId && firebaseConfig.appId);
}

/**
 * Firestore must NEVER be statically imported at module scope.
 *
 * Firestore's client SDK builds its wire-protocol type registry using
 * `protobufjs`, which generates and evaluates JavaScript from strings at
 * *import time* for performance (`new Function(...)`). Cloudflare Workers
 * disallows dynamic code generation entirely ("Code generation from strings
 * disallowed for this context"), so a plain `import { getFirestore } from
 * 'firebase/firestore'` at the top of this file crashes the Worker the
 * instant this module is evaluated — on every single request, including
 * pages that never touch Firestore, because Next.js evaluates
 * client-component modules during SSR too.
 *
 * The fix is to load `firebase/firestore` with a dynamic `import()` instead,
 * and only in the browser, and only lazily on first actual use — dynamic
 * imports are not evaluated until the `import()` call itself runs, so the
 * problematic module-level code never executes during server rendering.
 * Every real consumer in this app only ever calls this from inside a
 * 'use client' hook/effect/handler (never at module scope), so awaiting a
 * promise here is safe everywhere it's actually used.
 */
let dbInstance: Firestore | null = null;
let dbPromise: Promise<Firestore> | null = null;

export function getDb(): Promise<Firestore> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Firestore is only available in the browser.'));
  }
  if (dbInstance) return Promise.resolve(dbInstance);
  if (!dbPromise) {
    dbPromise = import('firebase/firestore').then(({ getFirestore }) => {
      dbInstance = getFirestore(firebaseApp);
      return dbInstance;
    });
  }
  return dbPromise;
}