import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pantun AI — Pengelas Tema Pantun Melayu",
  description: "Sistem AI untuk mengenal pasti tema pantun Melayu menggunakan SVM, TextCNN, dan MalayBERT.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ms" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
