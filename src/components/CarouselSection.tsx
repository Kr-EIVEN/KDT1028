"use client";
import { Swiper, SwiperSlide } from "swiper/react";
import { Autoplay, Navigation } from "swiper/modules";
import "swiper/css";
import "swiper/css/navigation";
import Image from "next/image";
import { useState } from "react";
import { useRouter } from "next/navigation";

type Props = {
  title: string;
  images: string[];
  icon?: string;
  perView?: number;
  gap?: number;
  autoplayDelay?: number;
};

export default function CarouselSection({
  title,
  icon,
  images,
  perView = 4,           // ✅ 한 슬라이드에 4장
  gap = 24,
  autoplayDelay = 3500,
}: Props) {
  const router = useRouter();
  const [liked, setLiked] = useState<string[]>([]);
  const [selectedImg, setSelectedImg] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleLike = (img: string) => {
    if (!isLoggedIn) {
      alert("로그인이 필요합니다.");
      router.push("/login");
      return;
    }
    setLiked((prev) =>
      prev.includes(img) ? prev.filter((i) => i !== img) : [...prev, img]
    );
  };

  const handleImageClick = (img: string) => {
    if (!isLoggedIn) {
      alert("로그인이 필요합니다.");
      router.push("/login");
      return;
    }
    setSelectedImg(img);
  };

  return (
    <section className="py-14">
      <div className="max-w-[1680px] mx-auto px-4"> {/* ✅ 폭 확장 (4장 × 400px + 여백) */}
        <div className="flex items-center justify-center gap-2 mb-8">
          {icon && <Image src={icon} alt="" width={36} height={36} />}
          <h2 className="text-2xl font-bold tracking-wide">{title}</h2>
        </div>

        {/* ✅ Swiper 슬라이드 */}
        <Swiper
          spaceBetween={gap}
          slidesPerView={4}           // ✅ 고정 4장
          modules={[Autoplay, Navigation]}
          navigation
          autoplay={{
            delay: autoplayDelay,
            disableOnInteraction: false,
          }}
          className="select-none overflow-hidden"  // ✅ 오버플로우 제거
        >
          {images.map((src, i) => (
            <SwiperSlide key={i} className="flex justify-center">
              <div
                className="relative overflow-hidden rounded-xl border group cursor-pointer"
                style={{ width: "400px", height: "400px" }} // ✅ 카드 크기 고정
                onClick={() => handleImageClick(src)}
              >
                <Image
                  src={src}
                  alt={`carousel-${i}`}
                  fill
                  className="object-cover transition duration-300 group-hover:brightness-75"
                />

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
              </div>
            </SwiperSlide>
          ))}
        </Swiper>

        <div className="text-right mt-4">
          <a
            className="inline-flex items-center gap-1 hover:underline text-gray-700"
            href="#"
          >
            Show More
            <span className="material-icons text-sm">
            </span>
          </a>
        </div>
      </div>

      {/* 팝업 모달 */}
      {selectedImg && (
        <div
          className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImg(null)}
        >
          <div
            className="bg-white rounded-xl max-w-md w-full p-5 shadow-lg relative"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="absolute top-2 right-3 text-gray-500 hover:text-black"
              onClick={() => setSelectedImg(null)}
            >
              ✕
            </button>

            <Image
              src={selectedImg}
              alt="selected"
              width={400}
              height={400}
              className="rounded-lg mb-4 object-cover"
            />

            <h3 className="text-lg font-semibold mb-2">사진 정보</h3>
            <p className="text-sm text-gray-600 mb-2">
              예시 설명: 이 사진은 PicShow에 업로드된 작품입니다.
            </p>
            <p className="text-xs text-gray-400">
              촬영일: 2025-10-30 ｜ 작성자: PicShow User
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
