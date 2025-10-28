"use client";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import Image from "next/image";

type Props = { title: string; icon?: string; images: string[] };

export default function CarouselSection({ title, icon, images }: Props) {
  return (
    <section className="py-14">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-center gap-3 mb-8">
          {icon && <Image src={icon} alt="" width={40} height={40} />}
          <h2 className="text-xl md:text-2xl font-semibold">{title}</h2>
          {icon && <Image src={icon} alt="" width={40} height={40} />}
        </div>

        <Swiper spaceBetween={24} slidesPerView={1} breakpoints={{
          640:{slidesPerView:2}, 1024:{slidesPerView:4}
        }} autoplay>
          {images.map((src, i)=>(
            <SwiperSlide key={i}>
              <div className="aspect-square rounded-xl overflow-hidden border">
                <Image src={src} alt={`carousel-${i}`} fill className="object-cover"/>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>

        <div className="text-right mt-4">
          <a className="inline-flex items-center gap-1 hover:underline" href="#">
            Show More <span className="material-icons">keyboard_double_arrow_right</span>
          </a>
        </div>
      </div>
    </section>
  );
}
