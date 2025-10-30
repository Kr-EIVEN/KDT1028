"use client";

import React, { useEffect, useRef, useState } from "react";

type Props = {
  title: string;
  icon?: string;
  images?: string[];
  perView?: number;          // 보이는 카드 수 (기본 4)
  gap?: number;              // 카드 간격(px)
  intervalMs?: number;       // 자동 슬라이드 간격
  transitionMs?: number;     // 슬라이드 속도
  className?: string;        // 섹션 외부 추가 클래스
  sectionBgClass?: string;   // 섹션 전체 배경 (예: bg-[#1f2937] rounded-3xl px-6 py-8)
  rounded?: string;          // 카드 라운드
  borderClass?: string;      // 카드 테두리
};

const clamp = (n: number, total: number) => ((n % total) + total) % total;

export default function CarouselSection({
  title,
  icon,
  images = [],
  perView = 4,
  gap = 24,
  intervalMs = 3500,
  transitionMs = 500,
  className = "",
  sectionBgClass = "",
  rounded = "rounded-md",
  borderClass = "border-[3px] border-gray-300/70",
}: Props) {
  const imagesCount = images.length;
  const extended = imagesCount ? [...images, ...images.slice(0, perView)] : [];

  const [index, setIndex] = useState(0);
  const [anim, setAnim] = useState(true);
  const [hover, setHover] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 자동 슬라이드
  useEffect(() => {
    if (hover || imagesCount <= perView) return;
    timerRef.current = setTimeout(() => setIndex((i) => i + 1), intervalMs);
    return () => timerRef.current && clearTimeout(timerRef.current);
  }, [index, hover, intervalMs, imagesCount, perView]);

  // 무한 루프 점프 (끝 → 처음)
  useEffect(() => {
    if (!imagesCount) return;
    if (index === imagesCount) {
      const t1 = setTimeout(() => {
        setAnim(false);
        setIndex(0);
      }, transitionMs);
      const t2 = setTimeout(() => setAnim(true), transitionMs + 20);
      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
      };
    }
  }, [index, imagesCount, transitionMs]);

  const goto = (i: number) => {
    if (!imagesCount) return;
    setAnim(true);
    setIndex(i);
  };
  const prev = () => goto(index - 1 < 0 ? Math.max(imagesCount - 1, 0) : index - 1);
  const next = () => goto(index + 1);

  // CSS 변수 (폭/이동량 계산)
  const cssVars: React.CSSProperties = {
    ["--per" as any]: perView,
    ["--gap" as any]: `${gap}px`,
    ["--idx" as any]: index,
  };

  // 이동 거리 계산 (gap 포함)
  const translate = `translateX(calc(-1 * var(--idx) * (400px + var(--gap))))`; // ✅ 400px 카드 기준 이동

  return (
    <section className={`py-10 ${sectionBgClass} ${className}`}>
      {/* 제목 */}
      <div className="flex items-center justify-center gap-2 mb-4">
        {icon && <img src={icon} alt="" className="h-6 w-6" />}
        <h2 className="text-2xl font-bold tracking-wide">{title}</h2>
      </div>

      {/* 슬라이드 뷰포트 */}
      <div
        className="relative overflow-hidden mx-auto"
        style={{
          ...cssVars,
          width: `calc(${perView} * 400px + ${(perView - 1)} * var(--gap))`, // 4장 너비 확정
        }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
  {/* 중앙 정렬용 mx-auto 적용됨 */}


        {/* 트랙 */}
        <div
          className="flex select-none will-change-transform"
          style={{
            gap: `${gap}px`,
            transform: translate,
            transition: anim ? `transform ${transitionMs}ms ease` : "none",
          }}
        >
          {extended.map((src, i) => (
            <div
              key={`${src}-${i}`}
              className={`relative shrink-0 overflow-hidden box-border ${rounded} ${borderClass}`}
              style={{
                width: "400px",       // ✅ 카드 폭 고정
                height: "400px",      // ✅ 카드 높이 고정 (정사각형)
              }}
            >
              <img
                src={src}
                alt={`img-${i}`}
                className="absolute inset-0 h-full w-full object-cover"
                draggable={false}
              />
            </div>
          ))}
        </div>

        {/* Prev / Next 버튼 */}
        {imagesCount > perView && (
          <>
            <button
              aria-label="Prev"
              onClick={prev}
              className="absolute left-3 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full px-3 py-2 shadow"
            >
              ‹
            </button>
            <button
              aria-label="Next"
              onClick={next}
              className="absolute right-3 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full px-3 py-2 shadow"
            >
              ›
            </button>
          </>
        )}
      </div>

      {/* 인디케이터 */}
      {imagesCount > perView && (
        <div className="mt-4 flex justify-center gap-2">
          {Array.from({ length: imagesCount }).map((_, i) => (
            <button
              key={i}
              onClick={() => goto(i)}
              aria-label={`Go to group ${i + 1}`}
              className={`h-2 w-2 rounded-full ${
                i === clamp(index, imagesCount) ? "bg-white" : "bg-white/40"
              }`}
            />
          ))}
        </div>
      )}
    </section>
  );
}
