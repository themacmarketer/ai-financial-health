// services/geminiService.ts
//
// Thin client-side wrapper. This file contains NO API key and NO direct calls
// to Google. Every request goes to our own /api/gemini serverless function,
// which holds the key server-side. The exported function signatures are
// unchanged, so no component code needed to change.

import type { AnalysisResult, AnalysisRequest, Source, TrainingScenario } from '../types';
import type { Content } from '@google/genai';
import { decode } from '../utils/audio';
import { DEMO_RESULT } from './demoData';

const ENDPOINT = '/api/gemini';

// Read a File into a base64 string so it can travel as JSON to the proxy.
const fileToBase64 = (file: File): Promise<{ data: string; mimeType: string }> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () =>
      resolve({
        data: (reader.result as string).split(',')[1],
        mimeType: file.type,
      });
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

// Shared POST helper with friendly error surfacing.
async function callProxy(action: string, payload: any): Promise<any> {
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, payload }),
  });
  if (!res.ok) {
    let message = 'The service is temporarily unavailable. Please try again.';
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      /* keep default message */
    }
    throw new Error(message);
  }
  return res.json();
}

export const getFinancialAnalysis = async (
  request: AnalysisRequest
): Promise<AnalysisResult> => {
  // FAST PATH: demo data stays fully client-side, no network call.
  if (request.isDemo) {
    await new Promise((resolve) => setTimeout(resolve, 800));
    if (request.guidanceMode === 'secular') {
      const { spiritualPerspective, ...secularResult } = DEMO_RESULT;
      return secularResult;
    }
    return DEMO_RESULT;
  }

  // Convert any attached File to base64 before sending.
  let filePayload: { data: string; mimeType: string } | null = null;
  if (request.file) {
    filePayload = await fileToBase64(request.file);
  }

  const payload = {
    statements: request.statements,
    industry: request.industry,
    companyType: request.companyType,
    country: request.country,
    guidanceMode: request.guidanceMode,
    userInfo: request.userInfo,
    file: filePayload,
  };

  const result = await callProxy('analysis', payload);
  return result as AnalysisResult;
};

export const getScenarioRecommendation = async (
  scenario: TrainingScenario,
  userChoice: string,
  userInput: string
): Promise<string> => {
  try {
    const { text } = await callProxy('scenario', {
      scenario,
      userChoice,
      userInput,
    });
    return text;
  } catch (error) {
    console.error('Error getting scenario recommendation:', error);
    return "I'm sorry, I was unable to generate a recommendation at this time. Please try again.";
  }
};

// Name kept as-is (original had a typo "Antswer") so imports still resolve.
export const getVAAntswer = async (
  question: string,
  history: Content[]
): Promise<string> => {
  try {
    const { text } = await callProxy('va', { question, history });
    return text;
  } catch (error) {
    console.error('Virtual Assistant Error:', error);
    return "I'm sorry, I'm having trouble connecting right now. Please try again later.";
  }
};

export const getSpeech = async (text: string): Promise<Uint8Array | null> => {
  try {
    const { audioBase64 } = await callProxy('speech', { text });
    if (audioBase64) {
      return decode(audioBase64); // back to Uint8Array for PrayerAudioPlayer
    }
    return null;
  } catch (error) {
    console.error('Error generating speech:', error);
    return null;
  }
};
