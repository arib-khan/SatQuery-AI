/**
 * SERVER-ONLY Firebase ID token verification — SERVER ONLY.
 *
 * This deliberately does NOT use the `firebase-admin` SDK. That SDK pulls in
 * `jwks-rsa`, which pulls in an old `jose` version whose `"workerd"` package
 * export condition points at a file that doesn't exist in the published
 * package. That's harmless on a normal Node server, but it makes the whole
 * dependency un-bundleable for Cloudflare Workers (OpenNext's Cloudflare
 * adapter bundles the server with esbuild targeting `workerd`, which hits
 * that broken export and fails with "Could not resolve \"jose\""). More
 * generally, `firebase-admin` relies on several Node-only APIs (gRPC, native
 * crypto internals, etc.) that don't exist in the Workers runtime at all, so
 * swapping the dependency instead of patching around it is the fix.
 *
 * Verifying a Firebase ID token does NOT require any service-account secret:
 * Firebase signs ID tokens with a key from a small, published, rotating set
 * of Google-hosted public keys. Verifying the signature against those public
 * keys (plus checking issuer/audience/expiry) is exactly what
 * `firebase-admin`'s `verifyIdToken` does under the hood, and it's safe to
 * reimplement client-side-readable-key verification here because a public
 * key can only verify signatures, never create them.
 *
 * This makes FIREBASE_ADMIN_CLIENT_EMAIL / FIREBASE_ADMIN_PRIVATE_KEY
 * unnecessary — only the (public) project id is needed, and it falls back to
 * NEXT_PUBLIC_FIREBASE_PROJECT_ID if a separate server-only copy isn't set.
 */
import { createRemoteJWKSet, jwtVerify } from 'jose';

const PROJECT_ID = process.env.FIREBASE_ADMIN_PROJECT_ID || process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID;

// Google's published, rotating public keys for Firebase Auth ID tokens.
// https://firebase.google.com/docs/auth/admin/verify-id-tokens#verify_id_tokens_using_a_third-party_jwt_library
const GOOGLE_SECURETOKEN_JWKS_URL = 'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';

// createRemoteJWKSet caches the fetched keys (and respects their HTTP cache
// headers) internally, so this is safe to reuse across requests/invocations
// without refetching Google's keys every time.
let jwks: ReturnType<typeof createRemoteJWKSet> | null = null;
function getJwks() {
  if (!jwks) {
    jwks = createRemoteJWKSet(new URL(GOOGLE_SECURETOKEN_JWKS_URL));
  }
  return jwks;
}

/**
 * Verifies the Firebase ID token sent from the client (Authorization: Bearer <token>)
 * and returns the trusted uid. Throws if the token is missing/invalid/expired.
 *
 * Implements Firebase's documented third-party JWT verification checklist:
 * algorithm, key match, issuer, audience, and expiry are all checked by
 * jose's `jwtVerify` itself; `sub` (non-empty) and `auth_time` (not in the
 * future) are checked explicitly below since jose has no opinion on them.
 */
export async function requireUid(authorizationHeader: string | null): Promise<string> {
  if (!authorizationHeader?.startsWith('Bearer ')) {
    throw new Error('Missing bearer token');
  }
  if (!PROJECT_ID) {
    throw new Error(
      'Firebase project id is not configured on the server. Set FIREBASE_ADMIN_PROJECT_ID (or NEXT_PUBLIC_FIREBASE_PROJECT_ID).'
    );
  }

  const idToken = authorizationHeader.slice('Bearer '.length);

  const { payload } = await jwtVerify(idToken, getJwks(), {
    issuer: `https://securetoken.google.com/${PROJECT_ID}`,
    audience: PROJECT_ID,
    algorithms: ['RS256'],
  });

  const uid = payload.sub;
  if (!uid || typeof uid !== 'string') {
    throw new Error('Invalid token: missing subject claim.');
  }

  const authTime = payload.auth_time;
  if (typeof authTime === 'number' && authTime * 1000 > Date.now()) {
    throw new Error('Invalid token: auth_time is in the future.');
  }

  return uid;
}