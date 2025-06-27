# Fertilizer Recommendation Web App - Project Document

## Overview

This project is a real-world deployment of a machine learning model trained on the Kaggle "Predicting Optimal Fertilizers" dataset. It provides a web interface for farmers and agricultural professionals to input environmental and soil parameters and receive suitable fertilizer recommendations.

---

## 1. Tech Stack

### Frontend

- **Framework**: React (Vite)
- **Styling**: Tailwind CSS
- **UI Components**:
  - Dropdowns for categorical fields (Soil Type, Crop Type)
  - Numeric inputs for environmental and soil data
  - Custom floating help boxes with links to explanation pages
  - Design theme: **Nature-inspired, minimal, modern**
    - Use soft **green shades** and **white backgrounds**
    - Maintain clean spacing, large padding, and subtle shadows
    - Font: Rounded sans-serif or modern serif (e.g., Inter, Nunito)

### Backend

- **Framework**: Flask
- **Model Format**: `.joblib` file
- **Logic**: Uses a predefined predictor class to load the model and return top fertilizer predictions

---

## 2. Input Fields and Design

### User Input Form

| Field               | Type     | Source/Help Page    |
| ------------------- | -------- | ------------------- |
| Temperature (°C)    | Numeric  | /how-to/temperature |
| Humidity (%)        | Numeric  | /how-to/humidity    |
| Moisture (%)        | Numeric  | /how-to/moisture    |
| Soil Type           | Dropdown | Predefined list     |
| Crop Type           | Dropdown | Predefined list     |
| Nitrogen (mg/kg)    | Numeric  | /how-to/nitrogen    |
| Phosphorous (mg/kg) | Numeric  | /how-to/phosphorous |
| Potassium (mg/kg)   | Numeric  | /how-to/potassium   |

Each input field will have a `?` icon that, when clicked, opens a floating info box with a brief description and a link to a full page.

---

### User Input Constraints

| Field               | Type     | Source/Help Page    | Constraints (Suggested Range)                                                                    |
| ------------------- | -------- | ------------------- | ------------------------------------------------------------------------------------------------ |
| Temperature (°C)    | Numeric  | /how-to/temperature | 20 – 40                                                                                          |
| Humidity (%)        | Numeric  | /how-to/humidity    | 45 – 75                                                                                          |
| Moisture (%)        | Numeric  | /how-to/moisture    | 20 – 70                                                                                          |
| Soil Type           | Dropdown | Predefined list     | Clayey, Sandy, Red, Loamy, Black                                                                 |
| Crop Type           | Dropdown | Predefined list     | Sugarcane, Millets, Barley, Paddy, Pulses, Tobacco, Ground Nuts, Maize, Cotton, Wheat, Oil seeds |
| Nitrogen (mg/kg)    | Numeric  | /how-to/nitrogen    | 0 – 50                                                                                           |
| Phosphorous (mg/kg) | Numeric  | /how-to/phosphorous | 0 – 50                                                                                           |
| Potassium (mg/kg)   | Numeric  | /how-to/potassium   | 0 – 25                                                                                           |


## 3. Backend API Design

### Endpoint

`POST /api/predict`

### Request Body

```json
{
    "Temparature": 30,
    "Humidity": 80,
    "Moisture": 20,
    "Soil Type": "Sandy",
    "Crop Type": "Wheat",
    "Nitrogen": 50, 
    "Potassium": 30, 
    "Phosphorous": 70
}
```

### Response

```json
{
    "predictions": "20-20 10-26-26 14-35-14"
}
```

---

## 4. Predictor Class (Python)

A sample predictor class (already defined) will handle:

- Loading the `.joblib` model
- Preprocessing the input
- Making top-3 fertilizer predictions

---

## 5. Explanation Pages

Each help page (`/how-to/<parameter>`) includes:

### Example: `/how-to/nitrogen`

- **What is it?** Nitrogen is a primary nutrient that supports leafy growth.

- **How to Measure:**

  - Use soil test kits (manual strips or digital probes)
  - Submit soil samples to a local agri-lab

- **Visual Aids:**

  - [Image of soil test kit](https://example.com/nitrogen-kit.jpg)
  - Sample soil report

- **YouTube Guide:**

  - [Watch how to test soil nitrogen](https://www.youtube.com/watch?v=2ZPqgPkhkE8)

---

### `/how-to/temperature`

- Use a digital thermometer or fetch via weather APIs
- [YouTube demo](https://www.youtube.com/watch?v=kC6t3Tqk3w0)

### `/how-to/humidity`

- Measure using humidity sensors or weather apps
- [YouTube demo](https://www.youtube.com/watch?v=ZbN5XOtJodk)

### `/how-to/moisture`

- Use a capacitive soil moisture sensor or tensiometer
- [YouTube demo](https://www.youtube.com/watch?v=TZxAXu3hP94)

### `/how-to/phosphorous`

- Test with soil lab kits or digital meters
- [YouTube demo](https://www.youtube.com/watch?v=Ep7uR8SJmm4)

### `/how-to/potassium`

- Measure using similar methods as nitrogen and phosphorous
- [YouTube demo](https://www.youtube.com/watch?v=kGiPZ5IgeYk)

---

## 6. Deployment

- **Local**:
  - Start backend with `flask run`
  - Start frontend with `npm run dev`
- **Cloud**: Railway, Render, or VPS

---
##

