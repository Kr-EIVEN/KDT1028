// app/page.tsx (또는 이 파일이 있는 경로의 메인 페이지)
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
      <div className="max-w-6xl mx-auto px-4">
        <CarouselSection title="인기게시물" icon="/assets/imgs/fire.png" images={인기}/>
      </div>
      <div className="max-w-6xl mx-auto px-4">
        <CarouselSection title="최신게시글" icon="/assets/imgs/newicon.png" images={최신}/>
      </div>
      <Footer />
    </>
  );
}
