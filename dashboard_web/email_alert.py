import os
import requests

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()

def send_critical_alert(recipient_email: str, run_id: str, feature: str,
                        ks_score: float, health: float):

    if not RESEND_API_KEY:
        print("[!] EMAIL SKIPPED: RESEND_API_KEY env var not set.")
        return

    html_content = f"""
    <html>
    <body style="font-family:'Courier New',monospace;background:#050608;color:#e3e2e5;padding:20px;">
        <div style="border-top:3px solid #d11f1f;background:#0f1112;padding:30px;max-width:600px;margin:0 auto;">
            <h2 style="color:#d11f1f;margin-top:0;letter-spacing:2px;">⚠️ CRITICAL_DRIFT_DETECTED</h2>
            <p style="color:#9aa4ad;font-size:14px;">
                ModelShift-Lite has detected severe data distribution drift in your live pipeline.
                Immediate review is recommended.
            </p>
            <hr style="border-color:#1f2329;margin:20px 0;">
            <table style="width:100%;color:#e3e2e5;border-collapse:collapse;font-size:14px;">
                <tr><td style="padding:8px 0;color:#9aa4ad;">RUN_ID:</td><td>{run_id}</td></tr>
                <tr><td style="padding:8px 0;color:#9aa4ad;">PRIMARY_FAULT_FEATURE:</td>
                    <td style="color:#d11f1f;font-weight:bold;">{feature}</td></tr>
                <tr><td style="padding:8px 0;color:#9aa4ad;">KS_STATISTIC:</td>
                    <td>{ks_score:.4f}</td></tr>
                <tr><td style="padding:8px 0;color:#9aa4ad;">SYSTEM_HEALTH:</td>
                    <td style="color:#d11f1f;font-weight:bold;">{health:.1f} / 100</td></tr>
            </table>
            <hr style="border-color:#1f2329;margin:20px 0;">
            <p style="color:#6b7785;font-size:11px;text-align:center;">
                [SYS.OP.AUTOMATED_DISPATCH]<br>
                Log in to your ModelShift Terminal to view datasets and download reports.
            </p>
        </div>
    </body>
    </html>
    """

    try:
        print(f"[~] EMAIL ATTEMPT via Resend API to {recipient_email}...")
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "from": "ModelShift <onboarding@resend.dev>",
                "to": ["ktams2530@gmail.com"],
                "subject": f"[CRITICAL] ModelShift Alert: Drift in '{feature}'",
                "html": html_content
            },
            timeout=15
        )
        if response.status_code in (200, 201):
            print(f"[✓] EMAIL DISPATCHED via Resend to {recipient_email}")
        else:
            print(f"[!] EMAIL FAILED: Resend returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[!] EMAIL FAILED: {e}")