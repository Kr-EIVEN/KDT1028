"use client";

import { useState } from "react";
import Image from "next/image";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";

export default function EventPage() {
  const [tab, setTab] = useState("ongoing");

  // 진행 중인 이벤트
  const ongoingEvents = [
    {
      title: "소중한 사람과의 추억을 공유하세요!",
      desc: "나의 일상을 담다",
      target: "모든 회원 대상",
      date: "2025.10.1 ~ 2025.10.31",
      img: "/assets/imgs/event_img01.png",
    },
    {
      title: "각자의 세상의 자연을 담아보세요!",
      desc: "푸른 하늘, 찬란한 순간을 담다",
      target: "모든 회원 대상",
      date: "상시 이벤트",
      img: "/assets/imgs/event_img02.png",
    },
    {
      title: "도전해서 인기 작가가 되어보세요!",
      desc: "제 1회 PicShow 사진 공모전",
      target: "모든 회원 대상",
      date: "2025.10.1 ~ 25.12.31",
      img: "/assets/imgs/event_img03.png",
    },
  ];

  // 종료된 이벤트 (현재 없음)
  const endedEvents: any[] = [];

  const list = tab === "ongoing" ? ongoingEvents : endedEvents;

  return (
    <main className="bg-white min-h-screen text-gray-900">
      <Navbar />

      <section className="max-w-6xl mx-auto px-4 py-24">
        {/* 제목 */}
        <h1 className="text-3xl font-bold mb-6">이벤트</h1>

        {/* 탭 메뉴 */}
        <div className="flex gap-6 mb-10 border-b border-gray-300">
          <button
            className={`pb-2 ${
              tab === "ongoing"
                ? "text-green-600 border-b-2 border-green-600 font-semibold"
                : "text-gray-500 hover:text-gray-900"
            }`}
            onClick={() => setTab("ongoing")}
          >
            진행중인 이벤트
          </button>
          <button
            className={`pb-2 ${
              tab === "ended"
                ? "text-green-600 border-b-2 border-green-600 font-semibold"
                : "text-gray-500 hover:text-gray-900"
            }`}
            onClick={() => setTab("ended")}
          >
            종료된 이벤트
          </button>
        </div>

        {/* 카드 리스트 or 안내 문구 */}
        {list.length > 0 ? (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {list.map((ev, i) => (
              <div
                key={i}
                className="bg-gray-50 rounded-2xl overflow-hidden shadow hover:shadow-lg transition border border-gray-200"
              >
                {/* 이미지 */}
                <div className="relative aspect-square w-full">
                  <Image
                    src={ev.img}
                    alt={ev.title}
                    fill
                    className="object-contain bg-white"
                  />
                </div>

                {/* 내용 */}
                <div className="p-5">
                  <p className="text-sm text-gray-500 mb-1">{ev.target}</p>
                  <h2 className="text-lg font-semibold mb-1">{ev.desc}</h2>
                  <p className="text-sm text-gray-500 mb-3">{ev.date}</p>
                  <p className="text-base font-medium">{ev.title}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          // 종료된 이벤트 없을 때 표시
          <p className="text-center text-gray-500 mt-20">
            현재 종료된 이벤트가 없습니다.
          </p>
        )}
      </section>

      <Footer />
    </main>
  );
}


