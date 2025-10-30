"use client";

import { Typography } from "@material-tailwind/react";

const CURRENT_YEAR = new Date().getFullYear();

export function Footer() {
  return (
    <footer className="py-10 border-t border-gray-200 bg-white">
      <div className="container mx-auto px-6">
        <Typography
          color="gray"
          className="text-center text-sm md:text-base font-medium text-gray-600"
        >
          © {CURRENT_YEAR} PicShow. All rights reserved.
        </Typography>
      </div>
    </footer>
  );
}

export default Footer;
