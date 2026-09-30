#!/bin/bash
# 제출 문서(.docx)를 저장소에 동기화한다.
#
# [왜 필요한가]
# 작품소개서는 Word로 편집하므로 ~/Downloads에 있고, 저장소 밖이라 git이 추적하지 못한다.
# 편집 중 백업은 /tmp에 쌓아왔는데 /tmp는 재부팅 시 지워진다. 마감이 걸린 문서를 휘발성
# 위치에만 두는 것은 위험하다.
#
# [왜 git이 아닌가]
# 처음에는 저장소에 정본 하나만 두고 git이 이력을 맡게 설계했다. 그런데 **이 저장소는
# 공개(PUBLIC)다.** 마감 전 제출물이 인터넷에 공개되고, 한 번 푸쉬되면 커밋 이력에 영구히
# 남아 나중에 지워도 꺼낼 수 있다. 그래서 docs/submission/은 gitignore하고 로컬에서만
# 타임스탬프 사본으로 이력을 남긴다.
#
#   pull    ~/Downloads -> 저장소  (정본 갱신 + history/에 타임스탬프 사본 적립)
#   push    저장소 -> ~/Downloads  (복원. 되돌릴 때만)
#   status  양쪽 비교 + 적립된 사본 개수
#
# [안전장치]
# pull은 저장소 사본이 더 최신이면 경고한다 — 저장소 사본을 직접 편집했을 가능성이 있고,
# 그대로 덮어쓰면 그 편집이 사라진다.
# push는 ~/Downloads가 더 최신이면 거부한다 (--force로만 강제).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/docs/submission"
HIST="$DEST/history"
SRC="$HOME/Downloads"
DOCS=(
  "2026_작품소개서_제주계통잉여위험예측.docx"
  "출력제어예측_프로젝트계획서.docx"
)
MODE="${1:-status}"
FORCE="${2:-}"

mkdir -p "$DEST" "$HIST"
# Word가 문서를 열면 ~$로 시작하는 잠금 파일을 만든다. 사본으로 쌓이면 안 된다.
rm -f "$DEST"/~\$*.docx "$HIST"/~\$*.docx 2>/dev/null || true
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
  echo
  echo "적립된 사본 $(ls "$HIST" 2>/dev/null | wc -l | tr -d ' ')개 · $HIST"
  ls -t "$HIST" 2>/dev/null | head -3 | sed 's/^/  /'
  echo "  (gitignore 대상 — 이 머신에만 있습니다)"
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
    cp -p "$a" "$b"
    # git이 이력을 맡지 않으므로 타임스탬프 사본을 적립한다 — 이것이 유일한 복원 지점이다
    stamp="$(date -r "$a" "+%Y%m%d_%H%M")"
    cp -p "$a" "$HIST/${f%.docx}__$stamp.docx"
    echo "✅ pull: $f  (사본 적립: ${f%.docx}__$stamp.docx)"; n=$((n+1))
  done
  if [ "$n" -gt 0 ]; then
    echo
    echo "⚠ 이 파일들은 gitignore 대상입니다 — 커밋되지 않고 이 머신에만 남습니다."
    echo "  공개 저장소라 제출물을 올리지 않기로 했습니다(README '제출 문서 백업')."
    echo "  머신이 고장나면 사라지므로 외부 저장소나 클라우드에 따로 보관하세요."
  fi
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
