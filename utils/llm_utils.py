import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # works locally

# On HuggingFace, key comes from Space Secrets
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set. Add it in Space Settings → Variables and Secrets")

client = Groq(api_key=api_key)
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are MedScan AI — a friendly, caring doctor friend who speaks casually and warmly, like a trusted friend who happens to be a doctor. You are helping patients in Pakistan who may not have easy access to healthcare.

Your personality rules:
- Talk like a friend, not a textbook. Use "you" directly, be conversational.
- Never use bullet point walls or long paragraphs — keep it tight and readable.
- Be warm and reassuring but honest when something is serious.
- Use Pakistani context — mention local foods, remedies, and free government facilities.
- Never give specific drug names or dosages — always say "see a doctor for medication".

Every response MUST follow this exact structure, keep each section SHORT:

🔍 What's happening: 1-2 sentences explaining what condition this likely is and why their symptoms match.

💊 Quick relief at home: 2 specific home remedies in 1 line each — practical, things available in Pakistani homes.

⚠️ If you ignore this: 2-3 sentences — honest consequences of leaving it untreated. Make them understand why this matters.

🚨 Go to hospital immediately if: 2-3 red flag symptoms listed in one line each — only the serious ones.

🏥 See a doctor: One line — what type of doctor or facility, mention free government option if available.

Keep total response under 280 words. Be warm, be real, be helpful."""


def explain_symptoms(
    symptom_text: str,
    bert_prediction: dict,
    rag_context: str = "",
    patient_language: str = "english"
) -> str:

    lang_note = (
        "Reply in simple Urdu mixed with English medical terms only."
        if patient_language == "urdu"
        else "Reply in simple conversational English."
    )

    context_section = (
        f"\nRelevant medical facts:\n{rag_context}\n"
        if rag_context else ""
    )

    top3_text = ", ".join([
        f"{x['label']} ({x['score']*100:.0f}%)"
        for x in bert_prediction['top_3']
    ])

    prompt = f"""{lang_note}

Patient says: "{symptom_text}"

AI detected: {bert_prediction['category']} ({bert_prediction['confidence']*100:.1f}% confidence)
Other possibilities: {top3_text}
{context_section}

Now respond as their friendly doctor friend using the exact structure from your instructions.
Do not add extra sections. Do not repeat the symptoms back. Get straight to the point warmly."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500,
    )
    return response.choices[0].message.content


def explain_report(
    extracted_text: str,
    detected_regions: list,
    patient_language: str = "english"
) -> str:

    region_summary = ", ".join(
        [r['class'] for r in detected_regions]
    ) if detected_regions else "full report"

    lang_note = (
        "Reply in simple Urdu mixed with English medical terms only."
        if patient_language == "urdu"
        else "Reply in simple conversational English."
    )

    prompt = f"""{lang_note}

Patient uploaded a medical report. Detected sections: {region_summary}

Report content:
{extracted_text}

Analyze this report as their friendly doctor friend using this structure:

🔍 What this report shows: 2 sentences — what the key findings mean in plain words.

📊 Values to note: highlight any abnormal values in 1 line each — say if HIGH or LOW and what it means.

💊 What to do now: 2-3 practical steps including one home remedy if relevant.

⚠️ If ignored: 2 sentences on consequences of leaving abnormal values unaddressed.

🚨 See a doctor urgently if: 2-3 specific red flags from this report.

Keep under 280 words. Be warm and clear."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500,
    )
    return response.choices[0].message.content


def test_groq_connection():
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say: MedScan AI ready"}],
            max_tokens=20
        )
        print("Groq connected:", r.choices[0].message.content)
        return True
    except Exception as e:
        print(f"Groq connection failed: {e}")

