persona = {
    "egos": [
      {
          "roleName": "Portfolio Manager",
          "mbti": "",
          "weightedScore": 0.50,
          "philosophicalTendency": "Idealism and Rationalism",
          "responsibility": "The ultimate decision-maker for the fund's investment strategy, managing asset allocation, risk, and overall portfolio performance.",
          "keySkillsets": [
              "Deep macroeconomic and market understanding",
              "Advanced quantitative and qualitative analysis",
              "Strong leadership and communication"
          ],
          "teamProportion": "One lead PM per fund or strategy, sometimes with co-PMs.",
          "decisionAuthority": "Final decision on all investment actions (buy/sell/hold)."
      },
        {
          "roleName": "Investment Analyst",
          "mbti": "",
          "weightedScore": 0.35,
          "philosophicalTendency": "Materialism and Empiricism",
          "responsibility": "Conducts fundamental research on companies and industries, builds financial models, and provides detailed investment recommendations.",
          "keySkillsets": [
              "Proficiency in financial statement analysis",
              "Expertise in valuation techniques",
              "Sector-specific market research and reporting"
          ],
          "teamProportion": "A team of several analysts (e.g., 2-5+) supporting one PM, often specializing by sector.",
          "decisionAuthority": "Provides strong recommendations and research to inform the PM's decision."
      },
        {
          "roleName": "Quantitative Analyst",
          "mbti": "",
          "weightedScore": 0.10,
          "philosophicalTendency": "Empiricism and Rationalism",
          "responsibility": "Develops quantitative models and trading strategies, identifies signals, and optimizes portfolio construction using data science and statistics.",
          "keySkillsets": [
              "Advanced statistical analysis and data science",
              "Programming skills (e.g., Python, R)",
              "Strong background in mathematics and computer science"
          ],
          "teamProportion": "Varies; can be a separate team or integrated, dependent on the fund's investment philosophy.",
          "decisionAuthority": "Provides data-driven insights and models; may have direct decision authority in quantitative funds."
      },
        {
          "roleName": "Trader",
          "mbti": "",
          "weightedScore": 0.05,
          "philosophicalTendency": "Pragmatism",
          "responsibility": "Executes investment decisions by placing trades on the market to achieve the best possible price and manage execution risk.",
          "keySkillsets": [
              "Expertise in market mechanics and order types",
              "Proficiency with trading platforms",
              "Analysis of market liquidity and price dynamics"
          ],
          "teamProportion": "One or more traders supporting one or multiple funds/PMs.",
          "decisionAuthority": "No strategic decision-making; responsible for tactical execution based on PM's orders."
      }
    ],
    "superego": {
        "roleName": "Chief Investment Officer (CIO)",
        "weightedScore": 1.00,
        "philosophicalTendency": "Stoicism",
        "responsibility": "Provides final approval or rejection for major investment decisions, ensuring they align with the firm’s long-term strategic vision, risk tolerance, and ethical framework. Acts as the ultimate check on the entire investment process.",
        "keySkillsets": [
            "A profound capacity for abstract and conceptual reasoning, reflecting a philosophical depth of thought.",
            "Masterful intuitive judgment and an uncommon sense for discerning qualitative value, akin to an artist's taste.",
            "Exceptional risk governance and management expertise, serving as a pillar of the firm’s stability.",
            "A proven track record of successful investment leadership across diverse market cycles.",
            "Ethical and principle-based decision-making with a confident, decisive demeanor."
        ],
        "teamProportion": "Typically one individual in this role per fund or investment firm.",
        "decisionAuthority": "Possesses the final, non-negotiable authority to approve or reject a trade, regardless of the Portfolio Manager's decision. This power is exercised for governance, not for routine trading."
    }
}

pipeline = {
    "investmentPipeline": [
        {
            "stepId": "01_IdeationAndInvestigation",
            "stepName": "Ideation & Initial Investigation",
            "actionOwner": [
                "Investment Analyst",
                "Quantitative Analyst"
            ],
            "responsibility": "To actively scout for potential investment opportunities and generate preliminary insights that align with the fund's mandate.",
            "inputs": [
                {
                    "type": "Tools",
                    "description": "Financial data terminals, market scanners, and quantitative models."
                },
                {
                    "type": "Materials",
                    "description": "Raw market data, company news, and macroeconomic reports."
                }
            ],
            "outputs": [
                {
                    "type": "Insight",
                    "description": "A preliminary investment hypothesis or a list of potential stock candidates."
                },
                {
                    "type": "Material",
                    "description": "Initial screening report or a brief summary of findings."
                }
            ]
        },
        {
            "stepId": "02_FormalResearchAndThesis",
            "stepName": "Formal Research & Investment Thesis",
            "actionOwner": [
                "Investment Analyst"
            ],
            "responsibility": "To conduct rigorous, in-depth research to build a comprehensive financial model and a compelling investment thesis.",
            "inputs": [
                {
                    "type": "Insight",
                    "description": "The preliminary investment hypothesis from Step 01."
                },
                {
                    "type": "Materials",
                    "description": "Detailed company filings, industry reports, and competitor analysis."
                }
            ],
            "outputs": [
                {
                    "type": "Material",
                    "description": "A detailed research report, including a financial model (e.g., DCF, relative valuation) and a formal recommendation (Buy/Sell/Hold)."
                },
                {
                    "type": "Action",
                    "description": "A presentation of the investment thesis to the entire investment team."
                }
            ]
        },
        {
            "stepId": "03_InvestmentCommitteeDecision",
            "stepName": "Investment Committee Vetting & Decision",
            "actionOwner": [
                "Portfolio Manager",
                "Investment Analyst",
                "Quantitative Analyst",
                "Trader",
                "Chief Investment Officer"
            ],
            "responsibility": "To collectively discuss and vote on the presented investment thesis. This step ensures full team alignment, with the CIO holding the final authority to approve or veto the decision.",
            "inputs": [
                {
                    "type": "Insight",
                    "description": "The detailed research report and recommendation from Step 02."
                },
                {
                    "type": "Information",
                    "description": "Verbal presentation and Q&A with the research analyst."
                }
            ],
            "outputs": [
                {
                    "type": "Material",
                    "description": "A formally approved trade order with specific parameters (e.g., security, quantity, price range)."
                },
                {
                    "type": "Action",
                    "description": "The directive to proceed to the final trading stage."
                }
            ]
        },
        {
            "stepId": "04_ExecutionAndMonitoring",
            "stepName": "Trade Execution & Portfolio Monitoring",
            "actionOwner": [
                "Trader"
            ],
            "responsibility": "To execute the approved trade with efficiency and minimal market impact. The Portfolio Manager then monitors the position against the thesis.",
            "inputs": [
                {
                    "type": "Material",
                    "description": "The approved trade order from the Investment Committee."
                },
                {
                    "type": "Tools",
                    "description": "Real-time trading platforms and market data feeds."
                }
            ],
            "outputs": [
                {
                    "type": "Material",
                    "description": "Trade confirmation and a final log of the transaction."
                },
                {
                    "type": "Insight",
                    "description": "Ongoing performance metrics and insights for the Portfolio Manager to monitor the position."
                }
            ]
        }
    ]
}

environment = {
    "investmentTeamEnvironment": {
        "informationSources": [
              {
                  "sourceName": "Company Financials & Filings",
                  "category": "Material",
                  "description": "Financial reports, annual reports, and regulatory filings from exchanges like the Shanghai and Shenzhen Stock Exchanges, and the CSRC."
              },
            {
                  "sourceName": "Macroeconomic Data",
                  "category": "Information",
                  "description": "Economic indicators, monetary policy announcements from the PBOC, and government policy changes."
              },
            {
                  "sourceName": "Industry Research",
                  "category": "Material",
                  "description": "Reports from third-party research firms, market news, and industry-specific publications."
              }
        ],
        "analyticalTools": [
            {
                "toolName": "Financial Data Terminals",
                "category": "Tool",
                "description": "Platforms like Bloomberg Terminal or Refinitiv Eikon for real-time market data, news, and analytics."
            },
            {
                "toolName": "Internal Financial Models",
                "category": "Tool",
                "description": "Proprietary models for valuation (e.g., DCF, relative valuation), risk analysis, and portfolio optimization."
            },
            {
                "toolName": "Statistical & Programming Software",
                "category": "Tool",
                "description": "Languages like Python or R used for quantitative analysis, back-testing strategies, and data visualization."
            }
        ],
        "requiredKnowledgeAndSkills": [
            {
                "skillsetCategory": "Market-Specific Knowledge",
                "details": [
                    "In-depth understanding of A-share market regulations and trading rules.",
                    "Knowledge of Chinese economic policy and its impact on sectors.",
                    "Awareness of specific market behaviors and investor sentiment in China."
                ]
            },
            {
                "skillsetCategory": "Technical Expertise",
                "details": [
                    "Advanced financial modeling and valuation techniques.",
                    "Proficiency in statistical and quantitative analysis.",
                    "Expertise in portfolio construction and risk management principles."
                ]
            },
            {
                "skillsetCategory": "Soft Skills",
                "details": [
                    "Critical thinking and problem-solving.",
                    "Effective communication for articulating investment theses.",
                    "Collaboration and teamwork within the investment team."
                ]
            }
        ]
    }
}
