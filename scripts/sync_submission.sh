#!/bin/bash
# 제출 문서(.docx)를 저장소에 동기화한다.
#
# [왜 필요한가]
# 작품소개서는 Word로 편집하므로 ~/Downloads에 있고, 저장소 밖이라 git이 추적하지 못한다.
# 편집 중 백업은 /tmp에 쌓아왔는데 /tmp는 재부팅 시 지워진다. 마감이 걸린 문서를 휘발성
# 위치에만 두는 것은 위험하다.
#
# [설계]
# 타임스탬프 사본을 저장소에 쌓지 않는다 — git이 이미 이력을 관리한다. 저장소에는 정본
# 하나만 두고, 동기화 후 커밋하면 그 커밋이 곧 복원 지점이 된다.
#
#   pull    ~/Downloads -> 저장소   (편집 후, 커밋 전에 실행)
#   push    저장소 -> ~/Downloads   (복원. 되돌릴 때만)
#   status  양쪽 비교
#
# [안전장치]
# pull은 저장소 사본이 더 최신이면 경고한다 — 저장소 사본을 직접 편집했을 가능성이 있고,
# 그대로 덮어쓰면 그 편집이 사라진다.
# push는 ~/Downloads가 더 최신이면 거부한다 (--force로만 강제).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/docs/submission"
SRC="$HOME/Downloads"
DOCS=(
  "2026_작품소개서_제주계통잉여위험예측.docx"
  "출력제어예측_프로젝트계획서.docx"
)
MODE="${1:-status}"
FORCE="${2:-}"

mkdir -p "$DEST"
h() { [ -f "$1" ] && shasum -a 256 "$1" | cut -c1-12 || echo "------------"; }
mt() { [ -f "$1" ] && date -r "$1" "+%m-%d %H:%M" || echo "     없음    "; }

case "$MODE" in
status)
  printf "%-46s %-14s %-14s %s\n" "문서" "~/Downloads" "저장소" "상태"
  for f in "${DOCS[@]}"; do
    a="$SRC/$f"; b="$DEST/$f"
    if [ ! -f "$a" ] && [ ! -f "$b" ]; then st="양쪽 없음"
    elif [ ! -f "$b" ]; then st="저장소에 없음 → pull 필요"
    elif [ ! -f "$a" ]; then st="~/Downloads에 없음 → push로 복원 가능"
    elif [ "$(h "$a")" = "$(h "$b")" ]; then st="일치"
    elif [ "$a" -nt "$b" ]; then st="편집됨 → pull 필요"
    else st="⚠ 저장소가 더 최신 — 확인 필요"
    fi
    printf "%-46s %-14s %-14s %s\n" "${f:0:44}" "$(mt "$a")" "$(mt "$b")" "$st"
  done
  ;;
pull)
  n=0
  for f in "${DOCS[@]}"; do
    a="$SRC/$f"; b="$DEST/$f"
    [ -f "$a" ] || { echo "건너뜀 (원본 없음): $f"; continue; }
    if [ -f "$b" ] && [ "$(h "$a")" = "$(h "$b")" ]; then echo "일치 (변경 없음): $f"; continue; fi
    if [ -f "$b" ] && [ "$b" -nt "$a" ] && [ "$FORCE" != "--force" ]; then
      echo "⚠ 저장소 사본이 더 최신입니다: $f"
      echo "  저장소 사본을 직접 편집했을 수 있습니다. 덮어쓰면 그 편집이 사라집니다."
      echo "  확인 후 --force로 다시 실행하세요."
      continue
    fi
    cp -p "$a" "$b"; echo "✅ pull: $f"; n=$((n+1))
  done
  [ "$n" -gt 0 ] && echo -e "\n커밋하세요 — 그 커밋이 복원 지점이 됩니다:\n  git add docs/submission && git commit -m 'docs: 제출 문서 동기화'"
  ;;
push)
  for f in "${DOCS[@]}"; do
    a="$SRC/$f"; b="$DEST/$f"
    [ -f "$b" ] || { echo "건너뜀 (저장소에 없음): $f"; continue; }
    if [ -f "$a" ] && [ "$a" -nt "$b" ] && [ "$FORCE" != "--force" ]; then
      echo "❌ ~/Downloads가 더 최신입니다: $f"
      echo "  덮어쓰면 편집 내용이 사라집니다. 먼저 pull하거나, 정말 되돌리려면 --force."
      continue
    fi
    cp -p "$b" "$a"; echo "✅ push(복원): $f"
  done
  ;;
*)
  echo "사용: bash scripts/sync_submission.sh [status|pull|push] [--force]"; exit 1 ;;
esac
