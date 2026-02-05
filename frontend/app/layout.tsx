import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Mayo Clinic STRIP AI',
  description: 'Stroke blood clot classification using deep learning',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
