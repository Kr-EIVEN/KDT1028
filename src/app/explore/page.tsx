"use client";
import Image from "next/image";
import Navbar from "@/components/navbar";
import Footer from "@/components/footer";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const categories = ["전체", "사람", "예술", "풍경", "동물", "음식", "산업", "건축"];

const allItems = [
  "/assets/imgs/carousel-1.JPG", "/assets/imgs/carousel-2.JPG", "/assets/imgs/carousel-3.jpg",
  "/assets/imgs/carousel-4.JPG", "/assets/imgs/carousel-5.JPG", "/assets/imgs/carousel-6.JPG",
  "/assets/imgs/carousel-7.jpg", "/assets/imgs/carousel-8.jpg", "/assets/imgs/carousel-9.jpg",
  "/assets/imgs/carousel-10.jpg", "/assets/imgs/carousel-11.jpg", "/assets/imgs/carousel-12.jpg",
  "/assets/imgs/carousel-13.JPG", "/assets/imgs/carousel-14.JPG", "/assets/imgs/carousel-15.jpg",
  "/assets/imgs/carousel-16.jpg", "/assets/imgs/carousel-17.jpg", "/assets/imgs/carousel-18.jpg",
];

export default function ExplorePage() {
  const router = useRouter();
  const [active, setActive] = useState("전체");
  const [items, setItems] = useState(allItems.slice(0, 12));
  const [liked, setLiked] = useState<string[]>([]);
  const [selectedImg, setSelectedImg] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [searchType, setSearchType] = useState("hashtag");
  const [searchQuery, setSearchQuery] = useState("");

  // ✅ 무한 스크롤
  useEffect(() => {
    const handleScroll = () => {
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 300) {
        setItems((prev) => [
          ...prev,
          ...allItems.slice(prev.length, prev.length + 8),
        ]);
      }
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // ❤️ 좋아요 기능
  const handleLike = (src: string) => {
    if (!isLoggedIn) {
      alert("로그인이 필요합니다.");
      router.push("/login");
      return;
    }
    setLiked((prev) =>
      prev.includes(src) ? prev.filter((i) => i !== src) : [...prev, src]
    );
  };

  // 📸 이미지 클릭 → 팝업
  const handleImageClick = (src: string) => {
    if (!isLoggedIn) {
      alert("로그인이 필요합니다.");
      router.push("/login");
      return;
    }
    setSelectedImg(src);
  };

  // 🔍 여러 해시태그 입력 (# 자동 추가)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (searchType === "hashtag" && e.key === " ") {
      e.preventDefault();
      setSearchQuery((prev) => (prev.endsWith(" ") ? prev : prev + " #"));
    }
  };

  // 🔍 검색 필터링 (임시)
  const filteredItems = items.filter((img) => {
    if (active === "전체") return true;
    return img.toLowerCase().includes(active);
  });

  return (
    <main className="min-h-screen bg-[#f2f2f2] text-gray-900 relative overflow-hidden">
      <Navbar />

      {/* 📦 전체 콘텐츠 박스 */}
      <div
        className="
          max-w-7xl mx-auto pt-20 mb-20 px-6 pb-10 bg-white 
          rounded-[30px] shadow-2xl ring-1 ring-gray-200/60 backdrop-blur-sm relative z-10
          before:content-[''] before:absolute before:top-0 before:left-0 before:w-full before:h-10
          before:bg-gradient-to-b before:from-gray-300/20 before:to-transparent before:rounded-t-[30px]
        "
      >
        {/* 탐색 헤더 */}
        <section className="flex flex-col items-center justify-center text-center pt-8 pb-6">
          <h1 className="text-4xl font-extrabold mb-3">탐색</h1>
          <p className="text-gray-600 text-base">
            카테고리를 선택하고 마음에 드는 사진을 찾아보세요.
          </p>
        </section>

        {/* 검색창 + 카테고리 */}
        <section className="flex flex-col items-center space-y-6">
          {/* 검색창 */}
          <div className="flex items-center gap-3 w-full max-w-4xl bg-white rounded-full shadow px-8 py-4 border border-gray-200">
            <select
              value={searchType}
              onChange={(e) => {
                setSearchType(e.target.value);
                setSearchQuery(""); // 검색 타입 바꿀 때 초기화
              }}
              className="bg-transparent text-gray-700 border-r pr-4 outline-none text-base"
            >
              <option value="hashtag"># 해시태그 검색</option>
              <option value="author">👤 작가 검색</option>
            </select>

            <input
              type="text"
              placeholder={
                searchType === "hashtag"
                  ? "#으로 시작하는 해시태그를 입력하세요..."
                  : "작가 이름을 입력하세요..."
              }
              value={searchQuery}
              onChange={(e) => {
                let value = e.target.value;

                if (searchType === "author") {
                  setSearchQuery(value);
                  return;
                }

                if (searchType === "hashtag" && !value.startsWith("#")) {
                  value = "#" + value.replace(/^#/, "");
                }
                setSearchQuery(value);
              }}
              onKeyDown={handleKeyDown}
              className="flex-grow bg-transparent outline-none px-4 text-gray-700 text-base"
            />
            <button className="text-gray-500 hover:text-gray-800 text-xl">
              🔍
            </button>
          </div>

          {/* 카테고리 버튼 */}
          <div className="flex flex-wrap justify-center gap-3 w-full">
            {categories.map((c) => (
              <button
                key={c}
                onClick={() => setActive(c)}
                className={`px-6 py-2.5 rounded-full border transition text-base ${
                  active === c
                    ? "bg-gray-900 text-white border-gray-900"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </section>

        {/* 📸 사진 그리드 */}
        <section className="mt-10">
          <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {filteredItems.map((src, i) => (
              <div
                key={i}
                className="group relative rounded-xl overflow-hidden border border-gray-200 shadow hover:shadow-lg transition cursor-pointer"
                onClick={() => handleImageClick(src)}
              >
                <div className="relative aspect-square">
                  <Image
                    src={src}
                    alt={`photo-${i}`}
                    fill
                    className="object-cover group-hover:scale-105 transition duration-300"
                  />
                </div>

                {/* ❤️ 하트 아이콘 */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleLike(src);
                  }}
                  className={`absolute bottom-3 right-3 text-2xl transition transform ${
                    liked.includes(src)
                      ? "text-red-500 scale-110"
                      : "text-white opacity-0 group-hover:opacity-100 group-hover:scale-110"
                  }`}
                >
                  {liked.includes(src) ? "❤️" : "🤍"}
                </button>

                {/* 해시태그 표시 */}
                <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/70 to-transparent">
                  <p className="text-sm text-white opacity-90">#{active}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <Footer />
    </main>
  );
}

