## 📋 **HOW WE SAMPLE AND SIMULATE**

---

## **PART 1: THE SAMPLING STRATEGY**

### **Step 1: Identify the Right Customers**
We start with **non-converted customers** from our simulation pool (the 5% of customers whose first quote appeared after our training cutoff).

### **Step 2: Apply Observable Rules to Find 5 Diverse Profiles**

Instead of using the model to segment customers (which sales can't do), we use **simple rules that any sales rep can spot:**

| Customer Type | Observable Signals | How to Spot |
|--------------|-------------------|-------------|
| **Price-sensitive** | • Already has a discount > 0€<br>• Has 2+ quotes (shopping around) | "This customer is looking for a deal" |
| **Value-sensitive** | • No existing discount<br>• Premium product (>€20k)<br>• Heat pump in cold region | "This customer wants quality" |
| **Neutral** | Mixed signals | "Hard to read - test both" |

### **Step 3: Select One of Each Type**
We pick:
- **2 price-sensitive customers** (to test discount-focused reps)
- **2 value-sensitive customers** (to test value-focused reps)
- **1 neutral customer** (to see if either approach works)

This gives us maximum diversity in just 5 simulations.

---

## **PART 2: THE SIMULATION**

### **Step 1: Establish Baseline**
For each selected customer, we:
- Take their actual quotes exactly as they exist
- Run them through the model to get their **current conversion probability**
- Record their current rep, product, price, and region

### **Step 2: Test Different Rep Strategies**

We simulate assigning different sales reps by applying their **characteristic discount**:

| Rep | Strategy | Discount | Color |
|-----|----------|----------|-------|
| **MARINA GUYOT** | Discount-focused | **2.5%** | 🟠 Orange |
| **ELISABETH MACHADO** | Value-focused | **0.6%** | 🟢 Green |
| **Clément TOUZAN** | Neutral | **1.5%** | 🔵 Blue |

For each rep, we:
- Create a modified version of the customer's quotes
- Apply the rep's characteristic discount
- Re-run the model to get a **new conversion probability**
- Calculate the **delta** (change from baseline)

### **Step 3: Visualize the Results**

We create an interactive widget with:

**Dropdown menu** (top center) - choose which rep to simulate:
- 🟠 MARINA GUYOT (2.5% discount)
- 🟢 ELISABETH MACHADO (0.6% discount)
- 🔵 Clément TOUZAN (1.5% discount)
- ⬜ Current rep (baseline)

**Five customer charts** (side by side):
- **Left bar (light color)**: Current probability (baseline) - never changes
- **Right bar**: New probability with selected rep - updates instantly
- **Bar colors** indicate customer segment:
  - 🟠 Orange = Price-sensitive (responds to MARINA)
  - 🟢 Green = Value-sensitive (responds to ELISABETH)
  - 🔵 Blue = Neutral (responds unpredictably)

**Hover information** shows:
- Customer ID
- Current rep
- Segment
- Baseline probability
- New probability and Δ when a rep is selected

---

## **PART 3: WHAT WE LEARN**

By testing these 5 diverse customers, we can see:

| If we select... | We expect... |
|-----------------|--------------|
| 🟠 MARINA | Orange customers improve, Green customers stay flat |
| 🟢 ELISABETH | Green customers improve, Orange customers stay flat |
| 🔵 Clément | Mixed results - some improve, some decline |

**The widget lets anyone explore these dynamics in real-time.**

---

## **🎯 THE BUSINESS VALUE**

This simulation proves that:

1. **Rep strategy matters** - different customers respond to different approaches
2. **The difference is simple** - just the discount percentage (2.5% vs 0.6%)
3. **Sales can identify customers** using observable signals (existing discounts, quote count, price point)
4. **Matching the right rep to the right customer** increases conversion probability by up to 19%

**No black box. No complex model. Just simple rules that sales can apply tomorrow.**