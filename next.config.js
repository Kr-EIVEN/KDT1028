/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",                     // 프론트에서 /api/... 호출 시
        destination: "http://127.0.0.1:5000/api/:path*", // Flask로 프록시 전달
      },
    ];
  },

  images: {
    // Flask가 반환하는 이미지도 표시 가능하게
    remotePatterns: [
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "5000",
      },
      {
        protocol: "http",
        hostname: "localhost",
        port: "5000",
      },
      {
        protocol: "https",
        hostname: "**", // 외부 이미지 허용 (기존 설정 유지)
      },
    ],
  },
};

module.exports = nextConfig;
