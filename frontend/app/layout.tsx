import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Klasifikasi Pantun Melayu | SVM Classifier",
  description: "Sistem pengklasifikasian tema pantun Melayu menggunakan algoritma SVM (Support Vector Machine). Masukkan pantun anda untuk mengenal pasti tema.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ms" suppressHydrationWarning>
      <body>
        <div className="bg-pattern" />
        {children}
      </body>
    </html>
  );
}
