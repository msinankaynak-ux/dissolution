#!/usr/bin/env bash
# DissolvA — her makineye oturunca çalıştır: tüm repoları GitHub'dan günceller.
# Diss-app/ altındaki .git içeren her klasörü fast-forward pull eder.
set -u
PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "=== DissolvA pull — $(date '+%Y-%m-%d %H:%M') ==="
for repo in "$PARENT"/*/; do
  [ -d "$repo/.git" ] || continue
  name="$(basename "$repo")"
  br="$(git -C "$repo" branch --show-current)"
  printf '%-22s [%s] ' "$name" "$br"
  if out=$(git -C "$repo" pull --ff-only 2>&1); then
    echo "$out" | tail -1
  else
    echo "!! PULL BAŞARISIZ — elle bak:"; echo "$out" | sed 's/^/    /'
  fi
done
echo "=== bitti ==="
