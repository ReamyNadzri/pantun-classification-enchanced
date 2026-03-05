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
      <body style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <div className="bg-pattern" />
        <div style={{ flex: 1 }}>
          {children}
        </div>
        <footer className="footer">
          <p>&copy; 2026 Owned by <span>AbeFiwanExpertStudio</span>. Hak Cipta Terpelihara.</p>
        </footer>
      </body>
    </html>
  );
}
