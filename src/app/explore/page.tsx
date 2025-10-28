"use client";
import Image from "next/image";
import Link from "next/link";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { useState } from "react";

const categories = ["전체", "풍경", "동물", "인물", "도시", "추상"];

const items = [
  "/assets/imgs/carousel-1.JPG", "/assets/imgs/carousel-2.JPG", "/assets/imgs/carousel-3.jpg",
  "/assets/imgs/carousel-4.JPG", "/assets/imgs/carousel-5.JPG", "/assets/imgs/carousel-6.JPG",
  "/assets/imgs/carousel-7.jpg", "/assets/imgs/carousel-8.jpg", "/assets/imgs/carousel-9.jpg",
];

export default function ExplorePage() {
  const [active, setActive] = useState("전체");

  return (
    <main className="bg-gray-950 min-h-screen text-white">
      <Navbar />

      {/* 헤더 (배경 통일) */}
      <section className="relative h-[36vh] w-full">
        <Image src="/bg.jpg" alt="Explore" fill priority className="object-cover opacity-80" />
        <div className="absolute inset-0 bg-black/40" />
        <div className="relative z-10 max-w-6xl mx-auto px-4 h-full flex flex-col items-center justify-center text-center">
          <h1 className="text-3xl md:text-5xl font-extrabold mb-3">탐색</h1>
          <p className="opacity-90">카테고리를 선택하고 마음에 드는 사진을 찾아보세요.</p>
        </div>
      </section>

      {/* 필터 */}
      <div className="max-w-6xl mx-auto px-4 py-6 flex flex-wrap gap-2 justify-center">
        {categories.map((c) => (
          <button
            key={c}
            onClick={() => setActive(c)}
            className={`px-4 py-2 rounded-full border ${active===c ? "bg-white text-gray-900" : "bg-white/10 hover:bg-white/20"}`}
          >
            {c}
          </button>
        ))}
      </div>

      {/* 카드 그리드 */}
      <div className="max-w-6xl mx-auto px-4 pb-14 grid gap-5 grid-cols-1 sm:grid-cols-2 md:grid-cols-3">
        {items.map((src, i) => (
          <Link href={`#`} key={i} className="group relative rounded-xl overflow-hidden border border-white/10 hover:border-white/30">
            <div className="relative aspect-square">
              <Image src={src} alt={`item-${i}`} fill className="object-cover group-hover:scale-105 transition" />
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/70 to-transparent">
              <p className="text-sm opacity-90">#{active} · Pic {i+1}</p>
            </div>
          </Link>
        ))}
      </div>

      <Footer />
    </main>
  );
}
