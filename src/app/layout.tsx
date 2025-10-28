import type { Metadata } from "next";
import "./globals.css";
import "swiper/css";
import Providers from "@/components/Providers";

export const metadata: Metadata = {
  title: "PicShow",
  description: "사진 공유 플랫폼",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
