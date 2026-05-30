#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

env_file="${ENV_FILE:-.env.production}"

if [[ "${1:-}" == "--env-file" ]]; then
  if [[ -z "${2:-}" ]]; then
    printf 'Usage: %s [--env-file path] {up|config|status|logs|down}\n' "$0" >&2
    exit 1
  fi
  env_file="$2"
  shift 2
fi

compose=(docker compose --env-file "$env_file" -f docker-compose.prod.yml)
action="${1:-up}"

if [[ $# -gt 0 ]]; then
  shift
fi

require_env_file() {
  if [[ ! -f "$env_file" ]]; then
    printf 'Missing %s. Start with: cp .env.production.example %s\n' "$env_file" "$env_file" >&2
    exit 1
  fi
}

prepare() {
  require_env_file
  ./scripts/check-prod-assets.sh
  "${compose[@]}" config >/dev/null
}

case "$action" in
  up)
    prepare
    "${compose[@]}" up --build -d
    "${compose[@]}" ps
    ;;
  config)
    prepare
    "${compose[@]}" config
    ;;
  status)
    require_env_file
    "${compose[@]}" ps
    ;;
  logs)
    require_env_file
    "${compose[@]}" logs --tail=200 -f "$@"
    ;;
  down)
    require_env_file
    "${compose[@]}" down
    ;;
  *)
    printf 'Usage: %s [--env-file path] {up|config|status|logs|down}\n' "$0" >&2
    exit 1
    ;;
esac
