from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supermarket Sales Analysis | IBM SkillsBuild Internship 2026</title>
    <style>
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: #F8FAFC;
            color: #0F172A;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 40px;
            max-width: 650px;
            width: 100%;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            border: 1px solid #E2E8F0;
            text-align: center;
        }
        .badge {
            background-color: #DBEAFE;
            color: #1E40AF;
            font-size: 13px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 9999px;
            display: inline-block;
            margin-bottom: 20px;
            letter-spacing: 0.025em;
        }
        h1 {
            color: #0F172A;
            font-size: 26px;
            font-weight: 800;
            margin: 0 0 12px 0;
        }
        p.subtitle {
            color: #475569;
            font-size: 15px;
            margin-bottom: 28px;
            line-height: 1.5;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            text-align: left;
            margin-bottom: 28px;
        }
        .info-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 16px;
            border-radius: 10px;
        }
        .info-card label {
            font-size: 12px;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            display: block;
            margin-bottom: 4px;
        }
        .info-card span {
            font-size: 14px;
            color: #1E40AF;
            font-weight: 700;
        }
        .btn-group {
            display: flex;
            gap: 12px;
            justify-content: center;
        }
        .btn {
            background-color: #2563EB;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .btn:hover {
            background-color: #1D4ED8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="badge">AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026</div>
        <h1>🛒 Supermarket Sales Analysis</h1>
        <p class="subtitle">AI-Powered Supermarket Sales Analysis, K-Means Customer Segmentation & Machine Learning System</p>
        
        <div class="info-grid">
            <div class="info-card">
                <label>Submitted By</label>
                <span>Medida Sri Venkata Praveen</span>
            </div>
            <div class="info-card">
                <label>Academic Branch</label>
                <span>B.Tech AIML</span>
            </div>
            <div class="info-card">
                <label>Repository</label>
                <span>Praveen-6163/ai-powered...</span>
            </div>
            <div class="info-card">
                <label>ML R² Score</label>
                <span>1.0000 (Random Forest)</span>
            </div>
        </div>
        
        <div class="btn-group">
            <a href="https://github.com/Praveen-6163/ai-powered-supermarket-sales-analysis" class="btn" target="_blank">View GitHub Repository</a>
        </div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode('utf-8'))
        return

app = handler
application = handler
