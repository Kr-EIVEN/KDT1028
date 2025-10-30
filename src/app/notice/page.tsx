"use client";

import Navbar from "@/components/navbar";
import Footer from "@/components/footer";

export default function NoticePage() {
  const notices = [
    {
      id: 4,
      category: "공지사항",
      title: "PicShow 리뉴얼 오픈! 더 쉽고 빠르게 사진을 공유하세요.",
      date: "2025.10.01",
    },
    {
      id: 3,
      category: "공지사항",
      title: "PicShow 개인정보처리방침 개정 안내 (2025년 9월 시행)",
      date: "2025.09.15",
    },
    {
      id: 2,
      category: "공지사항",
      title: "PicShow 서버 점검 안내 (7월 25일 오전 3시~5시)",
      date: "2025.07.20",
    },
    {
      id: 1,
      category: "공지사항",
      title: "PicShow 회원가입 기능 개선 안내",
      date: "2025.07.01",
    },
  ];

  return (
    <main className="bg-white min-h-screen text-gray-900">
      <Navbar />

      <section className="max-w-5xl mx-auto px-4 py-24">
        {/* 제목 */}
        <div className="mb-10 border-b pb-3">
          <h1 className="text-3xl font-bold">공지사항</h1>
          <p className="text-sm text-gray-500 mt-1">
            PicShow의 소식과 중요 사항을 빠르게 확인하세요.
          </p>
        </div>

        {/* 테이블 */}
        <div className="overflow-x-auto">
          <table className="w-full border-t border-gray-200">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="w-20 py-3 text-sm font-semibold text-gray-700 text-center">
                  번호
                </th>
                <th className="w-32 py-3 text-sm font-semibold text-gray-700 text-center">
                  분류
                </th>
                <th className="text-sm font-semibold text-gray-700 text-left">
                  제목
                </th>
                <th className="w-32 py-3 text-sm font-semibold text-gray-700 text-center">
                  등록일
                </th>
              </tr>
            </thead>
            <tbody>
              {notices.map((notice) => (
                <tr
                  key={notice.id}
                  className="border-b hover:bg-gray-50 transition"
                >
                  <td className="py-3 text-center text-gray-600">
                    {notice.id}
                  </td>
                  <td className="py-3 text-center text-green-700 font-medium">
                    {notice.category}
                  </td>
                  <td className="py-3 text-left text-gray-800">
                    {notice.title}
                  </td>
                  <td className="py-3 text-center text-gray-600">
                    {notice.date}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Footer />
    </main>
  );
}
