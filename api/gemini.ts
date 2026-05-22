// api/gemini.ts
//
// Server-side Gemini proxy. The GEMINI_API_KEY lives ONLY in this function's
// runtime environment (a Vercel environment variable). It is never sent to the
// browser. The frontend calls this endpoint; this endpoint calls Google.
//
// Routes four actions, matching the original geminiService.ts functions:
//   analysis -> getFinancialAnalysis | scenario -> getScenarioRecommendation
//   va -> getVAAnswer | speech -> getSpeech
//
// Includes a per-IP rate limiter and input size caps so a leaked endpoint URL
// cannot be turned into a free, unlimited Gemini relay.

import { GoogleGenAI, Type, Modality } from "@google/genai";

export const config = { runtime: "nodejs" };

const API_KEY = process.env.GEMINI_API_KEY;

// --- Rate limiter (in-memory, best-effort) --------------------------------
const WINDOW_MS = 60_000;
const MAX_REQUESTS = 10;
const hits: Map<string, { count: number; reset: number }> = new Map();
function rateLimited(ip: string): boolean {
  const now = Date.now();
  const rec = hits.get(ip);
  if (!rec || now > rec.reset) {
    hits.set(ip, { count: 1, reset: now + WINDOW_MS });
    return false;
  }
  rec.count += 1;
  return rec.count > MAX_REQUESTS;
}

const MAX_BODY_CHARS = 200_000;
const MAX_FILE_BYTES = 8 * 1024 * 1024;
const MODEL = "gemini-2.5-flash";

const analysisResultSchema = {
  type: Type.OBJECT,
  properties: {
    zScore: { type: Type.NUMBER },
    zone: { type: Type.STRING, enum: ["safe", "caution", "distress"] },
    zScoreModelUsed: { type: Type.STRING },
    zScoreModelDescription: { type: Type.STRING },
    zScoreModelBenchmarks: {
      type: Type.OBJECT,
      properties: {
        distress: { type: Type.STRING },
        caution: { type: Type.STRING },
        safe: { type: Type.STRING },
      },
      required: ["distress", "caution", "safe"],
    },
    explanation: { type: Type.STRING },
    turnaroundPlan: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          title: { type: Type.STRING },
          description: { type: Type.STRING },
          recommendations: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                description: { type: Type.STRING },
              },
              required: ["title", "description"],
            },
          },
        },
        required: ["title", "description", "recommendations"],
      },
    },
    sources: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: { title: { type: Type.STRING }, uri: { type: Type.STRING } },
        required: ["title", "uri"],
      },
    },
    helpfulResources: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: { name: { type: Type.STRING }, url: { type: Type.STRING } },
        required: ["name", "url"],
      },
    },
  },
  required: [
    "zScore", "zone", "zScoreModelUsed", "zScoreModelDescription",
    "zScoreModelBenchmarks", "explanation", "turnaroundPlan",
    "sources", "helpfulResources",
  ],
};

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }
  if (!API_KEY) {
    console.error("GEMINI_API_KEY not configured in environment");
    return res.status(500).json({ error: "Server is not configured." });
  }

  const ip =
    (req.headers["x-forwarded-for"] as string)?.split(",")[0]?.trim() ||
    "unknown";
  if (rateLimited(ip)) {
    return res
      .status(429)
      .json({ error: "Too many requests. Please wait a moment and retry." });
  }

  const rawLen = JSON.stringify(req.body || {}).length;
  if (rawLen > MAX_BODY_CHARS) {
    return res.status(413).json({ error: "Request too large." });
  }

  const ai = new GoogleGenAI({ apiKey: API_KEY });
  const { action, payload } = req.body || {};

  try {
    switch (action) {
      case "analysis":
        return res.status(200).json(await runAnalysis(ai, payload));
      case "scenario":
        return res.status(200).json({ text: await runScenario(ai, payload) });
      case "va":
        return res.status(200).json({ text: await runVA(ai, payload) });
      case "speech":
        return res.status(200).json(await runSpeech(ai, payload));
      default:
        return res.status(400).json({ error: "Unknown action." });
    }
  } catch (err: any) {
    console.error("Gemini proxy error (action=" + action + "):", err?.message || err);
    return res
      .status(502)
      .json({ error: "The analysis service failed. Please try again." });
  }
}

async function runAnalysis(ai: GoogleGenAI, payload: any) {
  const request = payload || {};

  const systemInstruction = "You are an expert in the corporate turnaround methodology of Dr. Michael Teng. This methodology is based on a medical analogy and consists of three phases:\n1. **Phase I: Surgery (Restructuring):** Focuses on immediate survival. Key actions include focusing on core competence, cost control, and improving cash flow. This phase requires a surgeon's skill.\n2. **Phase II: Resuscitation (Revitalization):** Focuses on growing the business. This involves developing the right product and price, aggressive marketing, and improving service quality to regain market share and profitability.\n3. **Phase III: Nursing (Rehabilitation):** Focuses on long-term health by building a strong and healthy corporate immune system. This involves cultivating a new corporate philosophy, fostering a culture of change, and empowering employees.\nYour analysis and recommendations must be structured around these three phases. You must also consider internal and external 'corporate viruses' when diagnosing issues.";

  const zScoreInstructions = "\n    3.  **Introduce the Altman Z-Score and its variants in your 'Z-Score Interpretation' section.** Explain that different models exist for different types of companies to ensure accuracy.\n\n    4.  **Select and Apply the Correct Altman Z-Score Model:** Use the table below to select the appropriate model based on the user's provided **Industry** and **Company Type**.\n\n        | Model       | Company Type                | Sector            | Equity Basis       | Distress Zone  | Caution Zone      | Safe Zone    |\n        |-------------|-----------------------------|-------------------|--------------------|----------------|-------------------|--------------|\n        | Z-Score     | Public                      | Manufacturing     | Market Value       | < 1.81         | 1.81 - 2.99       | > 2.99       |\n        | Z'-Score    | Private                     | Manufacturing     | Book Value         | < 1.23         | 1.23 - 2.90       | > 2.90       |\n        | Z''-Score   | Public or Private           | Non-Manufacturing | Book Value         | < 1.10         | 1.10 - 2.60       | > 2.60       |\n\n        *   **Z-Score Formula:** Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E (D = Market Value of Equity / Total Liabilities)\n        *   **Z'-Score Formula:** Z' = 0.717A + 0.847B + 3.107C + 0.420D + 0.998E (D = Book Value of Equity / Total Liabilities)\n        *   **Z''-Score Formula:** Z\" = 6.56A + 3.26B + 6.72C + 1.05D (D = Book Value of Equity / Total Liabilities; E term is removed)\n\n    5.  **Calculate the Z-Score Components:**\n        *   Working Capital = Current Assets - Current Liabilities\n        *   A = Working Capital / Total Assets\n        *   B = Retained Earnings / Total Assets\n        *   C = EBIT / Total Assets\n        *   D = (Use Market Value or Book Value of Equity as per model) / Total Liabilities\n        *   E = Sales / Total Assets (if applicable)\n\n    6.  **Determine the Distress Zone:** Based on the score and the thresholds from the table for the chosen model, determine if the company is in the 'safe', 'caution', or 'distress' zone and return this value in the 'zone' field.\n\n    7.  **Return Model Details:** You MUST return the details of the model you used.\n        *   Populate 'zScoreModelUsed' with the name of the model (e.g., \"Z''-Score\").\n        *   Populate 'zScoreModelDescription' with a brief explanation of why this model was chosen.\n        *   Populate 'zScoreModelBenchmarks' with the specific thresholds for the chosen model (e.g., distress: \"< 1.10\", caution: \"1.10 - 2.60\", safe: \"> 2.60\").\n  ";

  const coreAnalysisPrompt =
    "\n      You are an expert financial analyst and a senior partner at a top-tier business turnaround consultancy, channeling the methodology of Dr. Michael Teng. Your purpose is to provide an exceptionally deep, insightful, and actionable financial health assessment (minimum 3000 words) for Small and Medium Enterprise (SME) owners. Your tone is professional, authoritative, trustworthy, and encouraging.\n\n      **USER & COMPANY CONTEXT:**\n      *   **User Name:** " + (request.userInfo?.name || "N/A") +
    "\n      *   **Industry:** " + request.industry +
    "\n      *   **Company Type:** " + request.companyType +
    "\n      *   **Country:** " + (request.country || "Singapore") +
    "\n\n      **TASK:**\n      1.  **Holistic Document Analysis:** Treat the entire attached PDF document or pasted text as a knowledge base. Understand the company's business model, strategy, and market positioning.\n\n      " + zScoreInstructions +
    "\n\n      8.  **Analyze Key Turnaround Ratios:** You MUST calculate and analyze the following 15 financial ratios, grouped by their relevance to each turnaround phase.\n          *   **Surgery Phase Ratios (Immediate Crisis Management):** Current Ratio, Debt to Equity Ratio, Operating Cash Flow Ratio, Quick Ratio, Cash Ratio.\n          *   **Resuscitation Phase Ratios (Stabilization):** Return on Assets (ROA), Interest Coverage Ratio, Inventory Turnover Ratio, Receivables Turnover Ratio, Working Capital Ratio.\n          *   **Therapy Phase Ratios (Long-Term Growth):** Return on Equity (ROE), Gross Profit Margin, Asset Turnover Ratio, Profit Margin, Debt Service Coverage Ratio (DSCR).\n\n      9.  **Generate Extremely Comprehensive Analysis:** Based on the Z-Score and the 15 ratio analysis, generate a very detailed analysis in Markdown format for the 'explanation' field. This must be a minimum of 3000 words and MUST include these headings: '### Executive Summary', '### Z-Score Interpretation', '### Detailed Ratio Analysis' (with three sub-headings for each phase), and '### Risk Outlook'.\n\n      10. **Contextualize Analysis using Web Search:** Throughout your report, explicitly reference the company's industry ('" + request.industry + "') and country ('" + request.country + "'). When discussing the 15 key ratios, you MUST use the provided Google Search tool to find the most current industry and country-specific benchmarks for comparison. State the benchmarks you find in your analysis.\n\n      11. **Formulate a Turnaround Plan BASED on the Ratio Analysis:** Structure your recommendations into a comprehensive three-phase plan based on Dr. Michael Teng's methodology. The recommendations in each phase MUST directly address the weaknesses revealed by the 5 specific ratios you analyzed for that phase.\n\n      12. **Populate Helpful Resources:** You MUST populate the 'helpfulResources' field with EXACTLY the following 7 items: [{'name': 'M&A Centre', 'url': '#'}, {'name': 'Corporate Culture Centre', 'url': '#'}, {'name': 'Turnaround Centre', 'url': '#'}, {'name': 'Transformation Centre', 'url': '#'}, {'name': 'Change Management Centre', 'url': '#'}, {'name': 'Digital AI Centre', 'url': '#'}, {'name': 'Business Model Centre', 'url': '#'}].\n\n      **FINANCIAL STATEMENTS TEXT (if no file is attached):**\n      ---\n      " + request.statements + "\n      ---\n    ";

  const contents: any[] = [coreAnalysisPrompt];
  if (request.file && request.file.data && request.file.mimeType) {
    const approxBytes = (request.file.data.length * 3) / 4;
    if (approxBytes > MAX_FILE_BYTES) {
      throw new Error("Attached file exceeds size limit.");
    }
    contents.push({
      inlineData: { data: request.file.data, mimeType: request.file.mimeType },
    });
  }

  const analysisResponse = await ai.models.generateContent({
    model: MODEL,
    contents,
    config: {
      systemInstruction,
      temperature: 0.4,
      responseMimeType: "application/json",
      responseSchema: analysisResultSchema,
    },
    tools: [{ googleSearch: {} }],
  });

  const result: any = JSON.parse(analysisResponse.text);

  const groundingChunks =
    analysisResponse.candidates?.[0]?.groundingMetadata?.groundingChunks;
  const searchSources: any[] = [];
  if (groundingChunks) {
    for (const chunk of groundingChunks) {
      if (chunk.web) {
        searchSources.push({
          title: chunk.web.title || chunk.web.uri,
          uri: chunk.web.uri,
        });
      }
    }
  }
  const foundationalResources = [
    { title: "Corporate Turnaround: Nursing a sick company back to health", uri: "#" },
    { title: "Business Diagnosis", uri: "#" },
    { title: "Toolkit: Corporate Transformation to improve productivity and innovation", uri: "#" },
    { title: "Training Manual: Corporate Turnaround and Transformation Methodology", uri: "#" },
  ];
  const allSources = [...(result.sources || []), ...searchSources, ...foundationalResources];
  result.sources = Array.from(
    new Map(allSources.map((s: any) => [s.uri === "#" ? s.title : s.uri, s])).values()
  );

  if (request.guidanceMode === "faith-based") {
    try {
      const zoneImplication: Record<string, string> = {
        distress: "This score indicates a high probability of financial distress and bankruptcy in the near future. The situation requires immediate and drastic action to ensure survival.",
        caution: "This score indicates a gray area. The company is facing some financial pressure and could decline into distress if negative trends are not reversed.",
        safe: "This score indicates a strong financial position with a low probability of bankruptcy.",
      };
      const recommendationText = result.turnaroundPlan
        .map(
          (phase: any) =>
            "**" + phase.title + ":**\n" +
            phase.recommendations
              .map((rec: any) => "- " + rec.title + ": " + rec.description)
              .join("\n")
        )
        .join("\n\n");
      const summaryForPrayer =
        "\n              The company's financial analysis resulted in an Altman Z-Score of " +
        result.zScore.toFixed(2) +
        ", placing it in the '" + result.zone + "' zone.\n" +
        "              **Implication of the Zone:** " + zoneImplication[result.zone] +
        "\n\n              The following strategic recommendations have been proposed to address the situation:\n              " +
        recommendationText + "\n            ";
      const spiritualPrompt =
        "\n              You are a wise, Biblically-grounded Christian business advisor. Your task is to write a section titled '### Spiritual Perspectives on Secular Matters' that provides spiritual encouragement and a prayer based on a company's financial analysis.\n\n              **Analysis Summary & Recommendations:**\n              " +
        summaryForPrayer +
        "\n\n              **Instructions:**\n              1.  **Main Heading:** Start with the heading '### Spiritual Perspectives on Secular Matters'.\n              2.  **Scriptural Support (Part 1):** Before the prayer, write a section that connects the key secular recommendations to biblical principles. For each of the three turnaround phases (Surgery, Resuscitation, Nursing), pick 1-2 key recommendations from the summary and pair them with a relevant Bible verse and a short paragraph explaining the connection.\n              3.  **Prayer Content (Part 2):** After the scriptural support section, write a heartfelt prayer. The prayer text MUST begin with \"Our Heavenly Father,\".\n              4.  **Theological Foundation:** The prayer must be strictly Christian. You MUST use names like God, Jesus Christ, and Holy Spirit.\n              5.  **Be Specific and Relevant in Prayer:** The prayer should directly reference the company's situation as described in the summary, including the Z-Score zone and key challenges.\n              6.  **Cite Scripture in Prayer:** You MUST incorporate at least one additional Bible verse within the prayer itself for encouragement.\n              7.  **Tone:** Empathetic, encouraging, and reverent.\n              8.  **Closing:** You MUST end the prayer with the exact phrase: \"We pray all these in the name of Jesus Christ, Amen.\"\n              9.  **Length & Format:** The total length should be approximately 750 words. Return ONLY the markdown text.\n            ";
      const spiritualResponse = await ai.models.generateContent({
        model: MODEL,
        contents: spiritualPrompt,
      });
      result.spiritualPerspective = spiritualResponse.text;
    } catch (e) {
      console.error("Faith-based generation failed:", e);
      result.spiritualPerspective =
        "### Spiritual Perspective Not Available\n\nAn issue occurred while generating the faith-based prayer. The core financial analysis was successful and is available below.";
    }
  }

  return result;
}

async function runScenario(ai: GoogleGenAI, payload: any): Promise<string> {
  const { scenario, userChoice, userInput } = payload || {};
  const systemInstruction = "You are an expert business turnaround mentor and AI coach. Your tone is wise, encouraging, and highly practical. You are providing feedback within an interactive training module that follows a detailed financial analysis.";
  const prompt =
    "\n    **Training Scenario Analysis**\n\n    A business leader is facing the following scenario:\n    - **Scenario Title:** \"" +
    (scenario?.title) + "\"\n    - **Scenario Description:** \"" + (scenario?.description) +
    "\"\n    - **Their Chosen Action:** \"" + userChoice +
    "\"\n    - **Additional Context Provided by User:** \"" + (userInput || "No additional context provided.") +
    "\"\n\n    **Your Task:**\n    Provide a concise and actionable recommendation based on their choice. Your response must follow this structure:\n    1.  **Acknowledge and Validate:** Briefly acknowledge their chosen path.\n    2.  **Analysis (Pros & Cons):** Analyze the potential upsides (pros) and downsides (cons) of their chosen action in this specific scenario.\n    3.  **Actionable Recommendation:** Provide clear, numbered next steps (2-3 steps) they should take immediately.\n    4.  **Guiding Principle:** Conclude with a short, memorable piece of wisdom or a guiding principle relevant to the situation.\n\n    Format your entire response in Markdown. Use headings like \"### Analysis\" and \"### Recommended Next Steps\".\n  ";
  const response = await ai.models.generateContent({
    model: MODEL,
    contents: prompt,
    config: { systemInstruction },
  });
  return response.text;
}

async function runVA(ai: GoogleGenAI, payload: any): Promise<string> {
  const { question, history } = payload || {};
  const systemInstruction = "You are a helpful virtual assistant for the Corporate Turnaround Centre, specializing in Dr. Michael Teng's methodologies.\n  Your ONLY purpose is to answer questions related to the Centre's services, Dr. Teng's books, and the concepts of corporate turnaround (Surgery, Resuscitation, Nursing), financial health, Altman Z-Score, and the 8 Centres of Excellence.\n  Do NOT answer questions outside this scope. If asked about unrelated topics (like the weather, politics, recipes, etc.), politely state that you can only answer questions about corporate turnaround and the services of the Centre.\n  Keep your answers concise, helpful, and professional. You are here to guide users, not to perform financial analysis yourself.";
  const chat = await ai.chats.create({
    model: MODEL,
    config: { systemInstruction },
    history: history || [],
  });
  const response = await chat.sendMessage({ message: question });
  return response.text;
}

async function runSpeech(ai: GoogleGenAI, payload: any) {
  const { text } = payload || {};
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
      },
    },
  });
  const base64Audio =
    response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  return { audioBase64: base64Audio || null };
}
