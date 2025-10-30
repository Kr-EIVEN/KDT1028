"use client";

import { useState } from "react";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";

export default function CustomerCenter() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggleFAQ = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  const faqList = [
    {
      q: "📷 PicShow는 어떤 서비스인가요?",
      a: "PicShow는 사진 작가와 일반 사용자가 자신이 촬영한 사진을 공유하고, 서로의 작품을 감상하거나 후원할 수 있는 이미지 공유 플랫폼입니다. 누구나 쉽게 참여하고, 감상하고, 소통할 수 있는 공간을 목표로 합니다.",
    },
    {
      q: "📁 업로드한 사진은 어떻게 관리되나요?",
      a: "회원은 개인 갤러리에서 자신이 업로드한 사진을 자유롭게 관리할 수 있습니다. PicShow는 업로드된 이미지를 원본 품질을 유지하면서 웹 최적화 처리합니다.",
    },
    {
      q: "❤️ 좋아요 및 후원 기능은 어떻게 이용하나요?",
      a: "로그인한 사용자는 마음에 드는 사진에 ‘좋아요’를 누를 수 있으며, 작가에게 후원도 가능합니다. 후원 기능은 현재 베타 테스트 중입니다.",
    },
    {
      q: "🪶 무료로 이용 가능한가요?",
      a: "PicShow의 기본 기능(사진 보기, 좋아요, 검색 등)은 모두 무료로 제공됩니다. 향후에는 프리미엄 작가 전용 기능이 추가될 예정입니다.",
    },
    {
      q: "🙋 PicShow에 제안이나 문의는 어떻게 하나요?",
      a: "문의사항이나 제안이 있다면 ‘문의하기’ 버튼을 눌러 운영팀에 메시지를 남겨주세요. PicShow는 사용자 의견을 적극 반영하고 있습니다.",
    },
  ];

  return (
    <main className="min-h-screen bg-[#f9f9f9] text-gray-800 flex flex-col">
      <Navbar />

      {/* ✅ 전체 레이아웃 */}
      <section className="flex-grow max-w-6xl mx-auto px-8 py-28">
        {/* 제목 + 문의하기 버튼 */}
        <div className="flex justify-between items-center mb-12">
          <h1 className="text-3xl font-extrabold tracking-tight text-gray-900">
            고객센터 (FAQ)
          </h1>
          <button
            onClick={() => alert("문의 기능은 준비 중입니다!")}
            className="bg-gray-900 text-white text-sm px-6 py-2.5 rounded-md hover:bg-gray-700 transition"
          >
            문의하기
          </button>
        </div>

        {/* 검색창 */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 mb-12">
          <input
            type="text"
            placeholder="무엇을 도와드릴까요?"
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-gray-400 outline-none"
          />
        </div>

        {/* FAQ 목록 */}
        <div className="bg-white rounded-2xl shadow-md border border-gray-200 divide-y divide-gray-200">
          {faqList.map((faq, i) => (
            <div
              key={i}
              className="p-6 hover:bg-gray-50 transition cursor-pointer"
              onClick={() => toggleFAQ(i)}
            >
              {/* 질문 */}
              <div className="flex justify-between items-center">
                <p className="font-semibold text-lg">{faq.q}</p>
                <span className="text-gray-400 text-xl transition">
                  {openIndex === i ? "▲" : "▼"}
                </span>
              </div>

              {/* 답변 */}
              <div
                className={`overflow-hidden transition-all duration-300 ${
                  openIndex === i ? "max-h-48 mt-3" : "max-h-0"
                }`}
              >
                <p className="text-gray-600 text-sm leading-relaxed">
                  {faq.a}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <Footer />
    </main>
  );
}
