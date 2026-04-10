import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================
# ⚠️ REPLACE THESE WITH YOUR CREDENTIALS
# ==========================================
SENDER_EMAIL = os.environ.get("SMTP_EMAIL", "")
APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "") 
# ==========================================

def send_critical_alert(recipient_email: str, run_id: str, feature: str, ks_score: float, health: float):
    """Drafts and sends a cyberpunk-themed HTML email to the user."""
    
    subject = f"[CRITICAL] ModelShift Alert: Drift Detected in '{feature}'"

    # The HTML template for the email (styled to match your dashboard)
    html_content = f"""
    <html>
    <body style="font-family: 'Courier New', monospace; background-color: #050608; color: #e3e2e5; padding: 20px;">
        <div style="border-top: 3px solid #d11f1f; background-color: #0f1112; padding: 30px; border-left: 1px solid #1f2329; border-right: 1px solid #1f2329; border-bottom: 1px solid #1f2329; max-width: 600px; margin: 0 auto;">
            
            <h2 style="color: #d11f1f; margin-top: 0; letter-spacing: 2px;">⚠️ CRITICAL_DRIFT_DETECTED</h2>
            <p style="color: #9aa4ad; font-size: 14px;">ModelShift-Lite has detected severe data distribution drift in your live pipeline. Immediate review is recommended.</p>
            
            <hr style="border-color: #1f2329; margin: 20px 0;">
            
            <table style="width: 100%; color: #e3e2e5; border-collapse: collapse; font-size: 14px;">
                <tr>
                    <td style="padding: 8px 0; color: #9aa4ad;">RUN_ID:</td>
                    <td>{run_id}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #9aa4ad;">PRIMARY_FAULT_FEATURE:</td>
                    <td style="color: #d11f1f; font-weight: bold;">{feature}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #9aa4ad;">KS_STATISTIC:</td>
                    <td>{ks_score:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #9aa4ad;">SYSTEM_HEALTH:</td>
                    <td style="color: #d11f1f; font-weight: bold;">{health:.1f} / 100</td>
                </tr>
            </table>
            
            <hr style="border-color: #1f2329; margin: 20px 0;">
            
            <p style="color: #6b7785; font-size: 11px; text-align: center;">
                [SYS.OP.AUTOMATED_DISPATCH] <br>
                Log in to your ModelShift Terminal to view datasets and download reports.
            </p>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = f"ModelShift-Lite <{SENDER_EMAIL}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html'))

    try:
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() # Secure the connection
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[✓] EMAIL DISPATCHED: Critical alert sent to {recipient_email}")
    except Exception as e:
        print(f"[!] EMAIL FAILED: Could not send alert to {recipient_email}. Error: {e}")