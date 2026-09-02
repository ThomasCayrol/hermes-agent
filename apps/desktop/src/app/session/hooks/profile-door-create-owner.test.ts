import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Regression: "Session owner could not be resolved" on a BRAND-NEW session
// created from the project/branch view when the active gateway is a local
// profile-door secondary (multi-profile HotelOS setup: hotelos-cdp etc. are
// served by the desktop-spawned local primary, NOT a registry connection).
//
// The create used to record an owner ONLY when an explicit registry route
// existed. A profile-door create has no registry connectionId, so the fresh
// session resolved no owner until its first prompt persisted a DB row — and
// any session-scoped RPC in that window (tile resume before the runtime
// binding, a prompt on the fresh session) failed closed with
// SessionOwnerResolutionError. The owner IS knowable at create time: the bare
// profile of the door that minted it.

import { recordAmbientCreateOwner } from '@/app/session/hooks/use-session-actions'
import { $activeGatewayProfile, $newChatProfile, $profiles } from '@/store/profile'
import {
  _resetSessionOwnerHintsForTests,
  getSessionOwnerHint,
  knownSessionOwner,
  ownerLookupSessionRows,
  setSessions
} from '@/store/session'

beforeEach(() => {
  $profiles.set([{ name: 'default' }, { name: 'hotelos-cdp' }] as never)
  $activeGatewayProfile.set('hotelos-cdp')
  $newChatProfile.set('hotelos-cdp')
})

afterEach(() => {
  $profiles.set([])
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.clearAllMocks()
})

describe('profile-door session creation ownership (Mission Control new-session regression)', () => {
  it('records a profile-only owner for an ambient UNLISTED create on a non-default profile', () => {
    const recorded = recordAmbientCreateOwner('stored-door-1', $newChatProfile.get(), $activeGatewayProfile.get(), false)

    expect(recorded).toBe('hotelos-cdp')
    expect(getSessionOwnerHint('stored-door-1')).toMatchObject({ connectionId: '', profile: 'hotelos-cdp' })

    // No row exists yet (row persists on first prompt) — the owner ladder must
    // still resolve the BARE PROFILE so requestGatewayForProfile dials the
    // door that minted it (never a blind ambient fallback).
    expect(knownSessionOwner(ownerLookupSessionRows(), 'stored-door-1')).toBe('hotelos-cdp')
  })

  it('leaves LISTED ambient creates on the legacy row-owner path (no hint needed)', () => {
    // The optimistic listed row carries the bare profile; a hint would be a
    // redundant second owner record for the same door.
    expect(recordAmbientCreateOwner('stored-listed-1', 'hotelos-cdp', 'hotelos-cdp', true)).toBeNull()
    expect(getSessionOwnerHint('stored-listed-1')).toBeUndefined()
  })

  it('leaves default-profile creates on the legacy ambient path (no hint needed)', () => {
    $activeGatewayProfile.set('default')
    $newChatProfile.set(null)

    expect(recordAmbientCreateOwner('stored-default-1', null, 'default', false)).toBeNull()
    expect(getSessionOwnerHint('stored-default-1')).toBeUndefined()
  })

  it('records nothing without a stored session id (create failed)', () => {
    expect(recordAmbientCreateOwner(null, 'hotelos-cdp', 'hotelos-cdp', false)).toBeNull()
  })

  it('resolves the profile-only owner even when a row appears without connection tag', () => {
    recordAmbientCreateOwner('stored-door-2', 'hotelos-cdp', 'hotelos-cdp', false)
    // Backend row arrives (first prompt persisted it): no connection_id — the
    // profile matches the hint, so the owner is still deterministic.
    setSessions([
      {
        connection_id: undefined,
        id: 'stored-door-2',
        profile: 'hotelos-cdp',
        source: 'desktop'
      } as never
    ])

    expect(knownSessionOwner(ownerLookupSessionRows(), 'stored-door-2')).toBe('hotelos-cdp')
  })
})
