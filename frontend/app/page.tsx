'use client';

import { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mayo-clinic-strip-ai-production.up.railway.app';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'cfUCTA4DuThL2sVezKTQ-V6ZNELQtCoBWawIVNRRCyw';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'X-API-Key': API_KEY,
        },
      });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to classify image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Mayo Clinic STRIP AI
          </h1>
          <p className="text-xl text-gray-600">
            Stroke Blood Clot Classification System
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="border-4 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-indigo-400 transition">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    className="max-w-md max-h-64 rounded-lg mb-4"
                  />
                ) : (
                  <svg
                    className="w-24 h-24 text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                )}
                <span className="text-lg font-medium text-gray-700">
                  {file ? file.name : 'Click to upload image'}
                </span>
                <span className="text-sm text-gray-500 mt-2">
                  PNG, JPG, JPEG up to 16MB
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={!file || loading}
              className={`w-full py-4 px-6 rounded-xl text-white font-semibold text-lg transition
                ${
                  !file || loading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95'
                }`}
            >
              {loading ? 'Classifying...' : 'Classify Blood Clot'}
            </button>
          </form>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800 font-medium">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-8 space-y-4">
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">
                  Classification Result
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-gray-600 text-sm">Predicted Class</p>
                    <p className="text-3xl font-bold text-indigo-600">
                      {result.prediction}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      {result.prediction === 'CE'
                        ? 'Cardioembolic'
                        : 'Large Artery Atherosclerosis'}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600 text-sm">Confidence</p>
                    <p className="text-3xl font-bold text-green-600">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      {result.confidence > 0.9
                        ? 'Very High'
                        : result.confidence > 0.7
                        ? 'High'
                        : 'Moderate'}
                    </p>
                  </div>
                </div>

                {result.probabilities && (
                  <div className="mt-6">
                    <p className="text-gray-600 text-sm mb-2">Class Probabilities</p>
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>CE (Cardioembolic)</span>
                          <span>{(result.probabilities.CE * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${result.probabilities.CE * 100}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>LAA (Large Artery)</span>
                          <span>{(result.probabilities.LAA * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-purple-500 h-2 rounded-full"
                            style={{ width: `${result.probabilities.LAA * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800">
                  ⚠️ <strong>Research Use Only:</strong> This is not approved for clinical
                  diagnosis. Results should be reviewed by qualified medical professionals.
                </p>
              </div>
            </div>
          )}
        </div>

        <footer className="mt-8 text-center text-gray-600">
          <p>Mayo Clinic STRIP AI • Production Ready System • 2026</p>
        </footer>
      </div>
    </main>
  );
}
