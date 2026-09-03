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
    "anticholinergic", "cholinergic", "antipsychotic", "antidepressant",
    "anticoagulant", "anticoagulation", "antiplatelet", "antihypertensive",
    "antibiotic", "antiemetic", "analgesic", "sedative", "hypnotic", "diuretic",
    "statin", "statins", "opioid", "opiate", "benzodiazepine", "ssri", "snri",
    "inhibitor", "inhibitors", "agonist", "antagonist", "blocker", "blockers",
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
    "sick", "unwell", "heart", "kidney", "liver", "lung", "lungs", "brain",
    "blood", "bone", "skin", "joint", "muscle", "nerve", "vein", "artery",
    "bowel", "bladder", "stomach", "spine", "spinal", "complete", "partial",
    "transient", "persistent", "intermittent", "bilateral", "unilateral",
    "proximal", "distal", "superficial", "primary", "secondary", "benign",
    "malignant", "active", "inactive", "mixed", "probable", "definite",
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
        "ter", "ten",
    }
)
# "of" and "op" are deliberately absent. A particle is positive evidence of a
# person in `values.person_evidence`, and "of" would make `Lasting Power of
# Attorney` and `Activities of Daily Living` look like names.

#: Generational suffixes, which may end a value.
SUFFIXES: frozenset[str] = frozenset({"jr", "jnr", "sr", "snr", "ii", "iii", "iv"})

# --- Clinical head nouns ----------------------------------------------------
#
# A clinical noun phrase has a CLINICAL HEAD. `Sick Sinus Syndrome`, `Clock
# Drawing Test`, `Carer Strain Index`, `Chronic Kidney Disease` — the last word
# says what kind of thing the phrase names, and a person's name never ends in
# one of these. This is a small closed set of head nouns rather than an attempt
# to enumerate clinical English, which is why it generalises to phrases nobody
# listed: any `<Adjective> <Noun> Syndrome` is caught by `syndrome`.
#
# It is the rule that lets the person test be a REJECT test. Requiring positive
# proof of personhood instead — a gazetteer hit — leaked 49.3% of names, because
# a gazetteer of given names cannot be complete either. Rejecting what is
# positively clinical and accepting the rest fails in the safe direction for
# both invariants: an unrecognised clinical phrase is redacted (recoverable, and
# the placeholder still tells the model a field was there), while an
# unrecognised name is REMOVED rather than leaked.
CLINICAL_HEADS: frozenset[str] = frozenset(
    {
        # what a condition is called
        "syndrome", "disease", "disorder", "deficiency", "insufficiency",
        "failure", "attack", "infection", "infarction", "infarct", "thrombosis",
        "embolism", "haemorrhage", "hemorrhage", "stenosis", "sclerosis",
        "fibrosis", "atrophy", "degeneration", "dysfunction", "impairment",
        "injury", "lesion", "tumour", "tumor", "carcinoma", "neoplasm",
        "palsy", "paresis", "plegia", "pathy", "itis", "osis", "aemia", "emia",
        "bleeding", "ulcer", "oedema", "edema", "effusion", "necrosis",
        "arrest", "block", "murmur", "fibrillation", "flutter", "tachycardia",
        "bradycardia", "hypotension", "hypertension", "hypoxia", "sepsis",
        # what an instrument or measurement is called
        "test", "scale", "index", "score", "battery", "questionnaire",
        "inventory", "assessment", "examination", "screen", "profile",
        "grade", "stage", "category", "rating", "measure", "count", "level",
        "ratio", "load", "burden", "status", "reserve", "threshold",
        # what a document or process is called
        "summary", "report", "letter", "referral", "review", "plan", "record",
        "history", "impression", "findings", "recommendation", "discharge",
        "admission", "consultation", "attorney", "capacity",
        "living", "care", "package", "placement", "intake", "output",
    }
)


def has_clinical_head(tokens: list[str]) -> bool:
    """Whether this phrase ends in a word that names a clinical kind of thing.

    Also matches common morphological endings, so `Encephalopathy`,
    `Neuropathy`, `Osteoarthritis` and `Hypercalcaemia` are caught without being
    listed: a person's surname does not end in `-itis`, `-osis` or `-aemia`.
    """
    if not tokens:
        return False
    last = tokens[-1].lower().rstrip("s")
    if last in CLINICAL_HEADS:
        return True
    return any(
        last.endswith(ending)
        for ending in ("itis", "osis", "aemia", "emia", "pathy", "plegia", "paresis")
    )


# --- The positive signal ----------------------------------------------------
#
# `NOT_A_NAME` is a negative signal, and a negative signal loses this argument.
# `Peptic Ulcer Bleeding`, `Clock Drawing Test`, `Sick Sinus Syndrome` and
# `Lasting Power of Attorney` are the same SHAPE as `Harold Nkemdirim`, and the
# adversarial review destroyed 46 of 60 such phrases by pairing them with an
# orphan `Patient Name:` label. Enumerating clinical English is not a project
# that finishes.
#
# Given names are a much smaller and much more stable set than clinical
# vocabulary, so the discriminator is inverted: a bare line is accepted as a
# person ONLY on positive evidence — a title, a name particle, a non-Latin
# script, a given name from this list, or (in `layout`) proof that the document
# is a form with other fields that paired correctly.
#
# This list is not exhaustive and cannot be. Its gaps are covered by the form
# evidence, and what remains is recorded as a residual in the phase report.
GIVEN_NAMES: frozenset[str] = frozenset(
    {
        # Anglophone
        "james", "john", "robert", "michael", "william", "david", "richard",
        "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
        "anthony", "donald", "mark", "paul", "steven", "andrew", "kenneth",
        "george", "joshua", "kevin", "brian", "edward", "ronald", "timothy",
        "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
        "stephen", "jonathan", "larry", "justin", "scott", "brandon", "frank",
        "benjamin", "gregory", "samuel", "raymond", "patrick", "alexander",
        "jack", "dennis", "jerry", "tyler", "aaron", "henry", "douglas",
        "peter", "adam", "nathan", "zachary", "walter", "harold", "kyle",
        "carl", "arthur", "gerald", "roger", "keith", "jeremy", "terry",
        "lawrence", "sean", "albert", "joe", "ethan", "austin", "harry",
        "colin", "graham", "malcolm", "nigel", "trevor", "clive", "derek",
        "alan", "barry", "roy", "stanley", "leonard", "norman",
        "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa", "betty",
        "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
        "donna", "michelle", "carol", "amanda", "melissa", "deborah",
        "stephanie", "rebecca", "sharon", "laura", "cynthia", "amy",
        "kathleen", "angela", "shirley", "anna", "brenda", "pamela", "nicole",
        "ruth", "katherine", "samantha", "christine", "catherine", "virginia",
        "debra", "rachel", "janet", "emma", "carolyn", "maria", "heather",
        "diane", "julie", "joyce", "victoria", "kelly", "christina", "joan",
        "evelyn", "judith", "megan", "alice", "julia", "sophie", "olivia",
        "charlotte", "amelia", "isla", "ava", "grace", "freya", "florence",
        "fiona", "eileen", "maureen", "sheila", "gladys", "edith", "hilda",
        # Irish / Scottish / Welsh
        "aoife", "siobhan", "niamh", "sinead", "eoin", "cian", "declan",
        "seamus", "padraig", "ciara", "roisin", "hamish", "iain", "eilidh",
        "rhys", "dylan", "gareth", "owain", "carys", "bronwen",
        # Southern and Eastern Europe
        "jose", "juan", "carlos", "miguel", "antonio", "francisco", "manuel",
        "pedro", "javier", "sergio", "raul", "alberto", "carmen", "isabel",
        "pilar", "lucia", "elena", "rosa", "marta", "cristina", "beatriz",
        "giuseppe", "giovanni", "marco", "luca", "matteo", "francesca",
        "chiara", "giulia", "valentina", "alessandro", "lorenzo", "stefano",
        "joao", "ana", "sofia", "ines", "rui", "tiago",
        "hans", "klaus", "jurgen", "wolfgang", "helmut", "dieter", "gerhard",
        "ingrid", "ursula", "heidi", "monika", "petra", "sabine", "birgit",
        "pierre", "jacques", "michel", "philippe", "sylvie", "nathalie",
        "chantal", "monique", "genevieve", "olga", "irina", "svetlana",
        "natalia", "tatiana", "ekaterina", "dmitri", "vladimir", "sergei",
        "andrei", "mikhail", "nikolai", "aleksandr", "yelena", "anastasia",
        "piotr", "jakub", "agnieszka", "katarzyna", "malgorzata", "zofia",
        "ivan", "milos", "jelena", "dragan", "vesna", "nikos", "dimitris",
        "eleni", "yiannis",
        # South Asia
        "priya", "anjali", "deepa", "kavita", "meera", "sunita", "pooja",
        "neha", "shreya", "ananya", "aditya", "rahul", "vikram", "arjun",
        "sanjay", "rajesh", "amit", "suresh", "ramesh", "anil", "vijay",
        "ravi", "krishna", "lakshmi", "sita", "gita", "asha", "usha",
        "mohammed", "muhammad", "ahmed", "ali", "hassan", "hussain", "omar",
        "yusuf", "ibrahim", "khalid", "tariq", "imran", "bilal", "farhan",
        "fatima", "aisha", "zainab", "khadija", "amina", "sana", "hira",
        "nadia", "yasmin", "leila", "noor", "rania", "samira",
        # East and Southeast Asia
        "wei", "jing", "yan", "min", "hui", "ling", "mei", "xiu", "fang",
        "chen", "liang", "jun", "hao", "lei", "ming", "tao", "feng",
        "hiroshi", "takashi", "kenji", "yuki", "akira", "haruto", "sakura",
        "yuna", "minjun", "seoyeon", "jihoon", "jiwoo", "hyun", "sung",
        "nguyen", "linh", "trang", "huong", "thanh", "duc", "hoang",
        # Africa and the diaspora
        "chidinma", "chinelo", "ngozi", "adaeze", "amara", "ifeoma", "obiageli",
        "chukwuemeka", "emeka", "obinna", "ikechukwu", "nnamdi", "chibuzo",
        "kwame", "kofi", "yaw", "abena", "akosua", "ama", "afua",
        "thabo", "sipho", "nomsa", "zanele", "lerato", "bongani", "themba",
        "fatou", "aminata", "mariama", "ousmane", "moussa", "ibrahima",
        # Hebrew / Jewish
        "moshe", "avraham", "yitzhak", "yaakov", "shmuel", "chaim", "dovid",
        "rivka", "leah", "esther", "miriam", "chana",
        # Nordic
        "lars", "erik", "sven", "olav", "bjorn", "magnus", "anders", "nils",
        "astrid", "sigrid", "helga", "kari", "solveig", "annika",
    }
)



def _alternation(words: frozenset[str] | set[str]) -> str:
    """Longest-first alternation, so `mac` cannot shadow `macdonald`."""
    return "|".join(sorted((w.replace(".", r"\.") for w in words), key=len, reverse=True))


NOT_A_NAME_RE: str = _alternation(NOT_A_NAME)
PARTICLES_RE: str = _alternation(PARTICLES)
SUFFIXES_RE: str = _alternation(SUFFIXES)
TITLES_RE: str = _alternation(TITLES)
