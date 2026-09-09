# Autocw Customer Support Annotation & Labelling Guide

This guide establishes the standardized annotation protocol for evaluating the Autocw Customer Support Agent. It defines intent classification categories, boundary disambiguation rules, escalation criteria, and the 1–5 reply quality rubric.

---

## 1. Intent Taxonomy & Classification Rules

Every incoming message is assigned to exactly **one** primary intent label from the 8 mutually exclusive categories below.

### 1. `ORDER_STATUS`
- **Definition**: Inquiries regarding the current whereabouts, tracking number updates, estimated delivery date, or shipping confirmation of an existing order.
- **Key Indicators**: "where is my order", "tracking hasn't updated", "has it shipped", "when will it arrive", "check order status", order numbers `#123-4567`.
- **Exclusions**: If the customer specifically states the package is *already past the guaranteed delivery date* and expresses frustration with lateness, classify under `SHIPPING_DELAY`.

### 2. `REFUND_REQUEST`
- **Definition**: Explicit requests for money back, return shipping labels, billing dispute resolution, double charges, or cancellation reimbursements.
- **Key Indicators**: "refund", "return item", "charged twice", "money back", "cancelled order want money", "overcharged".
- **Precedence Rule**: If a customer mentions a delay or complaint *and* explicitly requests a refund ("Package is late, refund me now!"), classify as `REFUND_REQUEST` because financial resolution is the primary actionable demand.

### 3. `ACCOUNT_ISSUE`
- **Definition**: Authentication barriers, credential resets, account lockouts, security alerts, 2FA/OTP failures, or profile modifications.
- **Key Indicators**: "can't log in", "forgot password", "OTP not received", "verification code", "account locked", "email change", "unauthorized access".

### 4. `SHIPPING_DELAY`
- **Definition**: Grievances regarding shipments exceeding guaranteed delivery windows, transit halts, carrier delays, or missed delivery attempts.
- **Key Indicators**: "package is late", "missed delivery date", "delayed in transit", "supposed to arrive yesterday", "running 4 days late".
- **Disambiguation**: Neutral status requests ("When is it coming?") belong in `ORDER_STATUS`; grievance over elapsed delivery promises belongs in `SHIPPING_DELAY`.

### 5. `GENERAL_COMPLAINT`
- **Definition**: Dissatisfaction with service quality, rude agent behavior, physical package damage upon receipt, wrong item delivered, or poor product quality.
- **Key Indicators**: "item broken", "box was crushed", "terrible customer service", "unacceptable attitude", "wrong size delivered", "disgusted with your service".

### 6. `TECHNICAL_ISSUE`
- **Definition**: Functional failure of the digital storefront, website errors, mobile application crashes, checkout gateway exceptions, or broken UI elements.
- **Key Indicators**: "app crashed", "error 500", "button not clickable", "checkout won't load", "payment gateway failed", "website down".

### 7. `COMPLIMENT`
- **Definition**: Expressions of gratitude, praise, satisfaction, or recognition of excellent service.
- **Key Indicators**: "thank you", "thanks for the quick help", "great service", "kudos", "appreciate the assistance", "wonderful support".

### 8. `OTHER`
- **Definition**: Messages that are incomprehensible, purely greetings without intent ("hi"), completely out of scope ("do you sell cars?"), spam, or contradictory multi-intent queries without a discernible focus.

---

## 2. Escalation Protocol

Human escalation (`expected_escalation: true`) must be triggered whenever automated bot handling could create legal, financial, safety, or severe customer churn risks:

1. **Legal & Regulatory Threats**: Mentions of attorneys, lawyers, lawsuits, court, consumer protection bureaus (e.g. FTC, CFPB), or arbitration.
2. **Fraud & Criminal Claims**: Accusations of stolen packages, fraudulent credit card charges, driver theft, or police involvement.
3. **Severe Hostility & Abusive Language**: Explicit profanity directed at staff or uncontrollable anger.
4. **Explicit Human Demands**: "Speak to a human", "transfer me to a supervisor", "give me a live manager".
5. **High Financial Impact / Severe Loss**: Orders over $500 damaged or lost, or repeated delivery failures (>3 attempts).
6. **Ambiguity / Unclassifiable (`OTHER`)**: Queries where automated response is likely to frustrate the customer.

---

## 3. Reply Quality Rubric (1–5 Likert Scale)

For LLM-as-a-judge and human correlation evaluation:

| Score | Rating | Definition |
|:---:|:---:|:---|
| **5** | **Excellent** | Perfectly addresses the inquiry with empathetic tone, clear procedural instructions (e.g. tracking link or DM request), concise phrasing, and zero hallucinated policies. |
| **4** | **Good** | Accurately resolves the question with professional tone, minor non-critical omission of extra details, but completely safe and helpful. |
| **3** | **Acceptable** | Generic or formulaic response. Helpful enough to avoid customer frustration, but could be more tailored to the specific context. |
| **2** | **Poor** | Misses key aspects of the customer query, uses robotic or slightly dismissive tone, or suggests inappropriate troubleshooting steps. |
| **1** | **Unacceptable** | Irrelevant, completely ungrounded, hallucinated false policies/links, hostile tone, or dangerously inaccurate advice. |
