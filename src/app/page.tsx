"use client";

import { Navbar, Footer } from "@/components";
import Hero from "@/components/Hero";
import CarouselSection from "@/components/CarouselSection";

export default function Page() {
  const 인기 = [
    "/assets/imgs/carousel-1.JPG","/assets/imgs/carousel-2.JPG","/assets/imgs/carousel-3.jpg",
    "/assets/imgs/carousel-4.JPG","/assets/imgs/carousel-5.JPG","/assets/imgs/carousel-6.JPG",
    "/assets/imgs/carousel-7.jpg","/assets/imgs/carousel-8.jpg","/assets/imgs/carousel-9.jpg",
  ];
  const 최신 = [
    "/assets/imgs/carousel-10.jpg","/assets/imgs/carousel-11.jpg","/assets/imgs/carousel-12.jpg",
    "/assets/imgs/carousel-13.JPG","/assets/imgs/carousel-14.JPG","/assets/imgs/carousel-15.jpg",
    "/assets/imgs/carousel-16.jpg","/assets/imgs/carousel-17.jpg","/assets/imgs/carousel-18.jpg",
  ];

  return (
    <>
      <Navbar />
      <Hero />

      {/* ✅ 인기게시물 섹션 (부모 폭 제한 제거 + 중앙정렬) */}
      <div className="w-full flex justify-center px-4">
        <CarouselSection
          title="인기게시물"
          icon="/assets/imgs/fire.png"
          images={인기}
          perView={4}      // 4장
          gap={28}         // 카드 간격
          intervalMs={3500}
          transitionMs={500}
          // 선택: 섹션 배경을 주고 싶으면 아래 주석 해제
          // sectionBgClass="bg-[#1f2937] rounded-3xl px-8 py-12 shadow-xl"
        />
      </div>

      {/* ✅ 최신게시글 섹션 (동일 처리) */}
      <div className="w-full flex justify-center px-4 mt-10">
        <CarouselSection
          title="최신게시글"
          icon="/assets/imgs/newicon.png"
          images={최신}
          perView={4}
          gap={28}
          intervalMs={3500}
          transitionMs={500}
          // sectionBgClass="bg-[#1f2937] rounded-3xl px-8 py-12 shadow-xl"
        />
      </div>

      <Footer />
    </>
  );
}
