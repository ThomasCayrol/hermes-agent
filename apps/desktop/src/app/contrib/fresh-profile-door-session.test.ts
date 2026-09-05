import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// End-to-end regression for the Mission Control new-session ownership defect:
// a BRAND-NEW session created from the project/branch "+ New session" while
// the active gateway is a local profile-door secondary (hotelos-cdp) has no
// session row until its first prompt persists one. Before the fix the create
// recorded no owner (no registry connection to tag), so the first
// session-scoped RPC on the fresh session failed closed with
// SessionOwnerResolutionError. After the fix the ambient create records a
// profile-only owner hint, and the dispatcher routes the RPC to the
// profile-door socket that minted it.

const gatewayMocks = vi.hoisted(() => ({
  activeConnectionId: null as null | string,
  requestGatewayForAgent: vi.fn(async () => ({ routed: true })),
  requestGatewayForProfile: vi.fn(async () => ({ profiled: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  activeGatewayConnectionId: () => gatewayMocks.activeConnectionId,
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent,
  requestGatewayForProfile: gatewayMocks.requestGatewayForProfile
}))

const probe = vi.hoisted(() => ({ resolveSessionOwner: vi.fn(async () => undefined as unknown) }))
const sessionMocks = vi.hoisted(() => ({ requestSessionResume: vi.fn() }))

vi.mock('@/app/session/hooks/use-session-actions/utils', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  resolveSessionOwner: probe.resolveSessionOwner
}))

vi.mock('@/store/session', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestSessionResume: sessionMocks.requestSessionResume
}))

const { createSessionRpcDispatcher } = await import('./session-rpc-dispatcher')
const { $connectionsRegistry } = await import('@/store/connection-registry-state')
const { $profiles } = await import('@/store/profile')
const { _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions, setCronSessions, setMessagingSessions } =
  await import('@/store/session')
const { isSessionOwnerResolutionError } = await import('@/store/session-owner-resolution')
const { $sessionTiles } = await import('@/store/session-states')

function dispatcher(
  ambientRequest = vi.fn(async () => ({ ambient: true })),
  selectedStoredSessionId: null | string = null
) {
  return {
    ambientRequest,
    request: createSessionRpcDispatcher({
      ambientRequest: ambientRequest as never,
      runtimeIdByStoredSessionIdRef: { current: new Map([['stored-door-fresh', 'rt-fresh-door']]) },
      selectedStoredSessionIdRef: { current: selectedStoredSessionId },
      sessionStateByRuntimeIdRef: { current: new Map() }
    })
  }
}

beforeEach(() => {
  gatewayMocks.activeConnectionId = null // local profile-door secondary: NO registry connection
  $connectionsRegistry.set({ connections: [{ id: 'local' }] } as never)
  $profiles.set([{ name: 'default' }, { name: 'hotelos-cdp' }] as never)
  probe.resolveSessionOwner.mockResolvedValue(undefined)
})

afterEach(() => {
  $connectionsRegistry.set(null)
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  $sessionTiles.set([])
  $profiles.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  sessionMocks.requestSessionResume.mockReset()
  vi.clearAllMocks()
})

describe('fresh profile-door session RPC routing (Mission Control new-session)', () => {
  it('routes an unlisted fresh session to its profile door via the recorded profile-only hint', async () => {
    // What the create path now records the moment session.create returns a
    // stored id for an ambient (unrouted) profile-door create.
    setSessionOwnerHint('stored-door-fresh', { connectionId: '', profile: 'hotelos-cdp' })

    // No session row anywhere (fresh session, unlisted tile): the row rung
    // misses, but the hint must name the owner without a cross-profile probe
    // and without failing closed.
    const { ambientRequest, request } = dispatcher()

    await expect(request('prompt.submit', { session_id: 'rt-fresh-door', text: 'hello' })).resolves.toEqual({
      profiled: true
    })

    expect(gatewayMocks.requestGatewayForProfile).toHaveBeenCalledWith(
      'hotelos-cdp',
      'prompt.submit',
      { session_id: 'rt-fresh-door', text: 'hello' },
      undefined,
      undefined
    )
    expect(probe.resolveSessionOwner).not.toHaveBeenCalled()
    expect(ambientRequest).not.toHaveBeenCalled()
  })

  it('still fails closed (never blind-ambient) when a fresh session truly has no owner record', async () => {
    const { ambientRequest, request } = dispatcher()

    await expect(request('session.resume', { session_id: 'stored-no-owner' })).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    expect(ambientRequest).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForProfile).not.toHaveBeenCalled()
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()
  })
})
