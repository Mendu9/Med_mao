"""Words that are never part of a person's name.

A person's name cannot be recognised by shape alone. `Fairbanks` and `Amyloid`
are the same shape, and so are `Okonkwo` and `BRADYCARDIA`; the only thing that
separates them is vocabulary. Wave 10's A3 is exactly this:

    'GP: Dr Fairbanks Amyloid PET indication threshold question' -> 'GP: [NAME]'
    'Name: Mary Okonkwo BRADYCARDIA present should donepezil be avoided'
      -> 'Name: [NAME] should donepezil be avoided'

Both destroy the clinician's question, and the second destroys the exact
contraindication the model was being asked to reason about.

This is a stop list, and a stop list has a failure mode: a clinical word that is
not here can still be absorbed into a name value. That residual is real and is
recorded in the phase report. It is bounded, though, by the shape rules in
`values`, which already stop at the first lowercase word, the first token
carrying a digit, the first sentence terminator and the first known field label
— so this list only has to carry the words that survive all four, which means
capitalised or upper-case clinical vocabulary and capitalised common English.

Organised by category rather than alphabetically: a category is extensible by a
reader who knows the domain, an alphabetical list is only extensible by someone
who already knows what is missing.
"""
from __future__ import annotations

# Cognition, neurology and the eponyms a dementia letter is full of. Eponyms are
# the collision set for a name rule in this domain specifically: without them,
# "I reviewed Alzheimer's Disease guidance" redacts the disease.
_NEUROLOGY = {
    "alzheimer", "alzheimers", "parkinson", "parkinsons", "lewy", "braak",
    "scheltens", "creutzfeldt", "jakob", "huntington", "wernicke", "korsakoff",
    "broca", "wernickes", "binswanger", "pick", "picks", "down", "charcot",
    "hachinski", "rankin", "glasgow", "barthel", "addenbrooke", "addenbrookes",
    "boston", "trail", "stroop", "montreal", "mini", "mental", "moca", "mmse",
    "ace", "acer", "iii", "cdr", "gds", "npi", "fab", "rudas", "6cit", "tym",
    "dementia", "amnestic", "aphasia", "apraxia", "agnosia", "ataxia", "aphasic",
    "cognitive", "cognition", "memory", "amnesia", "delirium", "encephalopathy",
    "neurodegeneration", "atrophy", "hippocampal", "temporal", "frontal",
    "parietal", "occipital", "cortical", "subcortical", "cerebral", "cerebellar",
    "vascular", "ischaemic", "ischemic", "infarct", "haemorrhage", "hemorrhage",
    "stroke", "tia", "seizure", "epilepsy", "epileptic", "myoclonus", "tremor",
    "bradykinesia", "rigidity", "parkinsonism", "hydrocephalus", "meningitis",
    "encephalitis", "neuropathy", "myopathy", "radiculopathy", "neurology",
    "neurological", "neurologist", "psychiatry", "psychiatric", "geriatric",
    "geriatrics", "psychology", "psychometry", "behavioural", "behavioral",
}

# Imaging, laboratory and physiology. Almost all upper case, which is precisely
# the shape a name rule cannot otherwise reject.
_INVESTIGATIONS = {
    "mri", "ct", "pet", "spect", "eeg", "ecg", "ekg", "emg", "csf", "fdg",
    "dat", "datscan", "amyloid", "tau", "ptau", "abeta", "beta", "protein",
    "biomarker", "biomarkers", "lumbar", "puncture", "bloods", "fbc", "ue",
    "lft", "lfts", "tft", "tfts", "esr", "crp", "hba1c", "inr", "aptt", "gfr",
    "egfr", "vitamin", "b12", "folate", "tsh", "calcium", "sodium", "potassium",
    "creatinine", "urea", "albumin", "glucose", "cholesterol", "ldl", "hdl",
    "imaging", "scan", "scans", "radiology", "radiological", "histology",
    "biopsy", "pathology", "serology", "screen", "screening", "titre", "assay",
    "positive", "negative", "normal", "abnormal", "elevated", "reduced",
    "raised", "unremarkable", "equivocal", "pending", "awaited",
}

# Medications, the classes they belong to, and what a prescription line says.
_MEDICATIONS = {
    "donepezil", "rivastigmine", "galantamine", "memantine", "lecanemab",
    "donanemab", "aducanumab", "warfarin", "apixaban", "rivaroxaban",
    "edoxaban", "dabigatran", "aspirin", "clopidogrel", "atorvastatin",
    "simvastatin", "rosuvastatin", "amlodipine", "ramipril", "lisinopril",
    "bisoprolol", "atenolol", "digoxin", "furosemide", "metformin",
    "levothyroxine", "sertraline", "citalopram", "escitalopram", "fluoxetine",
    "mirtazapine", "trazodone", "quetiapine", "risperidone", "olanzapine",
    "haloperidol", "lorazepam", "diazepam", "zopiclone", "melatonin",
    "codeine", "paracetamol", "ibuprofen", "omeprazole", "lansoprazole",
    "prednisolone", "insulin", "gabapentin", "pregabalin", "levetiracetam",
    "lamotrigine", "sodium", "valproate", "carbamazepine", "phenytoin",
    "medication", "medications", "meds", "prescription", "prescribed", "dose",
    "doses", "dosage", "titrate", "titration", "tablet", "tablets", "capsule",
    "oral", "daily", "nocte", "mane", "bd", "tds", "qds", "prn", "po", "iv",
    "im", "sc", "mg", "mcg", "ml", "mmol", "bpm", "mmhg",
}

# Findings, symptoms, signs and the words a clinical narrative is built from.
_CLINICAL = {
    "bradycardia", "tachycardia", "arrhythmia", "fibrillation", "atrial",
    "sinus", "block", "syncope", "presyncope", "hypotension", "hypertension",
    "postural", "orthostatic", "falls", "fall", "gait", "balance", "mobility",
    "confusion", "confused", "agitation", "agitated", "aggression", "apathy",
    "depression", "depressed", "anxiety", "anxious", "psychosis", "psychotic",
    "hallucination", "hallucinations", "delusion", "delusions", "insomnia",
    "incontinence", "continence", "swallowing", "dysphagia", "weight", "loss",
    "appetite", "sleep", "fatigue", "pain", "infection", "sepsis", "pneumonia",
    "diabetes", "diabetic", "thyroid", "renal", "hepatic", "cardiac", "chest",
    "abdominal", "respiratory", "diagnosis", "diagnoses", "differential",
    "impression", "findings", "finding", "history", "examination", "assessment",
    "plan", "management", "treatment", "therapy", "prognosis", "referral",
    "follow", "review", "reviewed", "admission", "discharge", "summary",
    "outpatient", "inpatient", "clinic", "ward", "theatre", "consultation",
    "symptoms", "symptom", "signs", "sign", "onset", "duration", "progression",
    "progressive", "chronic", "acute", "stable", "unstable", "deterioration",
    "improvement", "baseline", "background", "comorbidity", "comorbidities",
    "allergy", "allergies", "nkda", "smoker", "alcohol", "independent",
    "dependent", "carer", "carers", "package", "care", "residential", "nursing",
    "capacity", "safeguarding", "advance", "decision", "dnacpr", "present",
    "absent", "untreated", "treated", "ongoing", "resolved", "recurrent",
}

# Capitalised common English. A sentence opening with one of these is prose, and
# an ALL-CAPS one is emphasis, not a surname.
_COMMON_ENGLISH = {
    "the", "and", "for", "with", "without", "from", "into", "onto", "upon",
    "this", "that", "these", "those", "there", "then", "than", "they", "them",
    "their", "his", "her", "she", "him", "who", "whom", "whose", "which",
    "what", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "very", "can", "will", "just", "should", "would", "could", "must", "may",
    "might", "shall", "does", "did", "done", "has", "have", "had", "been",
    "being", "was", "were", "are", "not", "nor", "but", "yet", "now", "new",
    "old", "current", "currently", "previous", "previously", "recent",
    "recently", "further", "additional", "possible", "probable", "likely",
    "unlikely", "consistent", "suggestive", "indicated", "indication",
    "threshold", "question", "questions", "please", "thank", "thanks", "dear",
    "yours", "sincerely", "faithfully", "regarding", "note", "notes", "letter",
    "report", "results", "result", "total", "score", "scores", "scale",
    "grade", "stage", "level", "levels", "range", "within", "above", "below",
    "left", "right", "bilateral", "mild", "moderate", "severe", "early", "late",
    "good", "poor", "fair", "well", "unwell", "yes", "no", "none", "nil",
    "first", "second", "third", "next", "last", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "night", "time", "times", "since",
    "until", "before", "after", "during", "while", "again", "also", "however",
    "although", "because", "therefore", "overall", "started", "stopped",
    "continued", "increased", "decreased", "unchanged", "confirmed", "denied",
}

# Honorifics. A title introduces a name; it is never the whole of one, so
# `Consultant: Dr` must not produce a `[NAME]` on its own.
TITLES = {
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "dame",
    "lord", "lady", "rev", "reverend", "sr", "br", "fr",
}

#: Every word that may not be a name token, lower-cased.
NOT_A_NAME: frozenset[str] = frozenset(
    _NEUROLOGY | _INVESTIGATIONS | _MEDICATIONS | _CLINICAL | _COMMON_ENGLISH | TITLES
)

#: Name particles. Lower case inside a name and nowhere else useful, so they are
#: allowed *between* name words but may never end a value.
PARTICLES: frozenset[str] = frozenset(
    {
        "van", "von", "de", "den", "der", "del", "della", "di", "da", "du",
        "la", "le", "el", "al", "bin", "ibn", "ben", "ap", "mac", "mc", "st",
        "ter", "ten", "op", "of",
    }
)

#: Generational suffixes, which may end a value.
SUFFIXES: frozenset[str] = frozenset({"jr", "jnr", "sr", "snr", "ii", "iii", "iv"})


def _alternation(words: frozenset[str] | set[str]) -> str:
    """Longest-first alternation, so `mac` cannot shadow `macdonald`."""
    return "|".join(sorted((w.replace(".", r"\.") for w in words), key=len, reverse=True))


NOT_A_NAME_RE: str = _alternation(NOT_A_NAME)
PARTICLES_RE: str = _alternation(PARTICLES)
SUFFIXES_RE: str = _alternation(SUFFIXES)
TITLES_RE: str = _alternation(TITLES)
