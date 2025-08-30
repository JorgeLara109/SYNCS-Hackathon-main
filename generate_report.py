# generate_report.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

file_path = "Full_Report.pdf"
c = canvas.Canvas(file_path, pagesize=A4)
width, height = A4

# Title
c.setFont("Helvetica-Bold", 18)
c.drawString(50, height - 50, "WasteWise Report")

# Add sections
c.setFont("Helvetica-Bold", 14)
c.drawString(50, height - 100, "Report Overview")
c.setFont("Helvetica", 12)
text = c.beginText(50, height - 120)
text.setLeading(18)

report_content = """
Report
One of the biggest challenges the modern urban cities will have to deal with will be climate change, resource scarcity and an evolving economic landscape. Our goal is to develop predictive models that analyse multiple resource usage processes under certain conditions. The quantification of this methods will help us in the future to approach problems more effectively. By leveraging machine learning models like k-Nearest Neighbors (kNN), Random Forest, and Linear Regression, we can move from abstract questions to concrete, data-informed decisions, directly tackling the three pivotal questions for the cities of the future.

1.	Climate change demands infrastructure that is not only efficient but also resilient and adaptive. Our models could help us to take justified solutions dynamically. For instance, a Random Forest model can analyze historical data on electricity usage, correlating it with factors like temperature, humidity, building density, and green space. This allows city planners to simulate the impact of specific interventions.
For example, a model might help us understand that the increase in n% of green areas in a certain location will lead us to a reduction of m% of cooling energy during summer.
The data could also help us understand the power usage in certain locations according to income. It is widely recognized that household electronics in lower-income neighborhoods often suffer from substandard quality, which means that the power usage of such devices may be inefficient or inconsistent. Lower-quality appliances often lack energy-saving features, leading to higher electricity consumption despite limited usage. According to this data, we could identify high-consumption, low-income areas and implement targeted state-sponsored electronics programs. One practical initiative could involve replacing outdated, inefficient light bulbs with energy-saving LED alternatives. This not only reduces household electricity bills but also contributes to broader sustainability goals by lowering overall energy demand.
2.	For resources like water and electricity, optimization is key to managing growth without waste. Linear Regression models are excellent for identifying the primary drivers of consumption. By analyzing data on water usage against factors like population density, industrial activity, and precipitation levels, we can pinpoint inefficiencies and predict future demand with high accuracy. For example, a linear model can identify non-revenue water (leakage) accounts for a significantly larger portion of usage in older districts than previously estimated, and that demand is most sensitive to small industrial users. This means that according to the data given by the models we could prioritize a pipe replacement program in the identified districts and develop conservation programs targeted specifically at small businesses, ensuring resources are directed where they will have the greatest impact.
3.	The rise of remote work and the gig economy shifts energy and transit patterns away from traditional central business districts. A kNN model can classify neighborhoods based on usage patterns, identifying areas that behave similarly. This helps understand the new geography of work. For example, the analysis could show that suburban neighborhood "X" has electricity and internet usage patterns (high daytime consumption) that closely resemble those of a known central business district, indicating it has become a significant hub for remote workers. This means that the city can invest in bolstering broadband infrastructure in neighborhood "X" and consider deploying shared, flexible co-working spaces to support this new cluster of workers, fostering local economic activity and reducing the need for long commutes.
In conclusion, our project quantifies resource usage processes that will help us to take justified qualitative decisions about new policies and infrastructure investments. The future of urban living will be built on data.
"""

for line in report_content.splitlines():
    text.textLine(line)
c.drawText(text)

# Save PDF
c.showPage()
c.save()
print(f"PDF saved to {file_path}")
