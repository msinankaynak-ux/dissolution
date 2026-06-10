#!/usr/bin/env bash
# DissolvA — tüm repoların GitHub'a göre durumu (ahead/behind, kirli dosya).
set -u
PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "=== DissolvA status — $(date '+%Y-%m-%d %H:%M') ==="
for repo in "$PARENT"/*/; do
  [ -d "$repo/.git" ] || continue
  name="$(basename "$repo")"; br="$(git -C "$repo" branch --show-current)"
  git -C "$repo" fetch --quiet 2>/dev/null || true
  ab=$(git -C "$repo" rev-list --left-right --count "origin/$br...$br" 2>/dev/null | awk '{print "behind "$1", ahead "$2}')
  dirty=$(git -C "$repo" status --porcelain | wc -l | tr -d ' ')
  printf '%-22s [%-5s] %s | kirli dosya: %s\n' "$name" "$br" "${ab:-takip yok}" "$dirty"
done
