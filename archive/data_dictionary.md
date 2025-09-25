# Data Dictionary for Mental Health Lifestyle Dataset

| Column                      | Description                              | Type    | Example Values         |
|-----------------------------|------------------------------------------|---------|-----------------------|
| Country                     | Country of respondent                    | String  | Brazil, Australia     |
| Age                         | Age in years                             | Integer | 18, 25, 40            |
| Gender                      | Gender identity                          | String  | Male, Female, Other   |
| Exercise Level              | Physical activity level                  | String  | Low, Moderate, High   |
| Diet Type                   | Main diet pattern                        | String  | Vegan, Keto, etc.     |
| Sleep Hours                 | Average sleep per night                  | Float   | 6.5, 7.2              |
| Stress Level                | Self-reported stress                     | String  | Low, Moderate, High   |
| Mental Health Condition     | Diagnosed condition                      | String  | None, Anxiety, etc.   |
| Work Hours per Week         | Weekly work hours                        | Integer | 40, 55                |
| Screen Time per Day (Hours) | Daily screen time                        | Float   | 4.5, 7.0              |
| Social Interaction Score    | Social activity (scale 1–10)             | Float   | 5.5, 8.0              |
| Happiness Score             | Self-reported happiness (scale 1–10)     | Float   | 6.5, 9.0              |
| Age Group                   | Age bin (engineered feature)             | String  | 18-25, 26-35, etc.    |

**Note:** `Age Group` will be added via feature engineering.
