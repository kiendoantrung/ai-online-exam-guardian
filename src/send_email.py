import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_warning_email(msv, warnings):
    sender_email = ""
    receiver_email = ""
    password = ""

    message = MIMEMultipart("alternative")
    message["Subject"] = "Warning Alert"
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create the HTML content
# Create the HTML content with CSS styling
    html = f"""
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
          }}
          .container {{
            width: 80%;
            margin: 20px auto;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          }}
          h2 {{
            color: #333333;
          }}
          p {{
            font-size: 16px;
            color: #555555;
          }}
          ul {{
            list-style-type: none;
            padding: 5px;
          }}
          .alert-act {{
            margin-top: 20px;
            font-size: 14px;
            color: red;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>Warning Alert for Student Code: {msv}</h2>
          <p class="alert-act">Please take the necessary actions.</p>
          <p>The warning count has reached the limit. Below is the list of warning cases:</p>
          <ul>
    """

    for warning in warnings:
        if warning:
            html += f"""
            <li>{warning}</li>
            """
    html += """
          </ul>
        </div>
      </body>
    </html>
    """

    part = MIMEText(html, "html")
    message.attach(part)


    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.send_message(message)
