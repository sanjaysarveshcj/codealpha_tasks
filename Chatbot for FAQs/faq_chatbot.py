import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import datetime
from collections import deque

class TechSupportChatbot:
    def __init__(self, max_history=10):   
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.faqs = self._init_tech_support_faqs()
        self.questions = list(self.faqs.keys())
        self.answers = list(self.faqs.values())
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
        self.conversation_history = deque(maxlen=max_history)
        
        self.follow_ups = self._generate_tech_follow_ups()
        
        self.tech_terms = {
            'wifi': 'WiFi',
            'bluetooth': 'Bluetooth',
            'cpu': 'CPU',
            'gpu': 'GPU',
            'lan': 'LAN',
            'ip': 'IP',
            'dns': 'DNS',
            'ssl': 'SSL',
            'url': 'URL',
            'vpn': 'VPN'
        }
        
        self.sentiment_responses = {
            'very_negative': [
                "I understand your frustration. Let me help resolve this issue right away.",
                "I'm sorry you're having such a difficult time. Let's fix this together.",
                "I apologize for the inconvenience. I'll do my best to help you."
            ],
            'negative': [
                "I'll help you sort this out.",
                "Let's work together to resolve this.",
                "I understand your concern. Here's what we can do:"
            ],
            'neutral': [
                "Here's what I found:",
                "Let me help you with that.",
                "I can assist you with this."
            ],
            'positive': [
                "Great question! Here's the information:",
                "I'm happy to help! Here's what you need:",
                "Excellent! Let me show you how:"
            ]
        }

    def _init_tech_support_faqs(self):    
        return {
            "Why is my internet connection slow?": """
                Here are common solutions for slow internet:
                1. Restart your router and modem
                2. Check for background downloads
                3. Run a speed test at speedtest.net
                4. Check for WiFi interference
                5. Consider using a wired connection
                Contact your ISP if the issue persists.""",
            
            "How do I reset my WiFi password?": """
                To reset your WiFi password:
                1. Access your router's admin panel (typically 192.168.1.1 or 192.168.0.1)
                2. Log in with admin credentials
                3. Find wireless settings
                4. Change the password
                5. Save changes and reconnect devices""",
            
            "Why does my WiFi keep disconnecting?": """
                Common solutions for WiFi disconnection issues:
                1. Check router placement and distance
                2. Update router firmware
                3. Change WiFi channel to reduce interference
                4. Update network adapter drivers
                5. Check for conflicting devices
                6. Consider a WiFi extender for better coverage""",
            
            "How do I set up a VPN?": """
                Steps to set up a VPN:
                1. Choose a VPN service provider
                2. Download and install the VPN client
                3. Launch the application
                4. Log in with your credentials
                5. Select a server location
                6. Connect and verify your new IP address""",
            
            "My computer is running slowly": """
                Try these steps to improve performance:
                1. Close unnecessary programs
                2. Check Task Manager for resource usage
                3. Run disk cleanup and defragmentation
                4. Scan for malware
                5. Consider upgrading RAM or switching to SSD
                6. Disable startup programs
                7. Update Windows and drivers""",
            
            "How do I update my drivers?": """
                To update your drivers:
                1. Open Device Manager
                2. Right-click the device
                3. Select 'Update driver'
                4. Choose automatic or manual update
                For graphics cards, use manufacturer software (NVIDIA/AMD).""",
            
            "Blue screen error appears regularly": """
                Steps to resolve Blue Screen of Death (BSOD):
                1. Note down the error code
                2. Update all drivers
                3. Run Windows Memory Diagnostic
                4. Check for Windows updates
                5. Scan for malware
                6. Check hardware connections
                7. Review recent software changes""",
            
            "Windows won't boot properly": """
                Try these solutions for boot issues:
                1. Enter Safe Mode
                2. Run Startup Repair
                3. Check disk for errors (chkdsk)
                4. Restore to previous point
                5. Scan for malware in Safe Mode
                6. Repair/Reset Windows if necessary""",
            
            "How do I protect against malware?": """
                Essential malware protection steps:
                1. Install reputable antivirus software
                2. Keep systems updated
                3. Use strong passwords
                4. Avoid suspicious downloads
                5. Enable firewall
                6. Regular system scans
                7. Use two-factor authentication
                8. Be cautious with email attachments""",
            
            "How do I create strong passwords?": """
                Password best practices:
                1. Use at least 12 characters
                2. Mix uppercase, lowercase, numbers, and symbols
                3. Avoid personal information
                4. Use different passwords for each account
                5. Consider a password manager
                6. Enable two-factor authentication
                7. Change passwords regularly""",
            
            "My printer isn't working": """
                Printer troubleshooting steps:
                1. Check power and connections
                2. Verify printer is set as default
                3. Clear print queue
                4. Check for paper jams
                5. Reinstall printer drivers
                6. Run printer troubleshooter
                7. Check ink/toner levels
                8. Update firmware""",
            
            "Computer makes strange noises": """
                Diagnose computer noises:
                1. Identify the source (fan, hard drive, etc.)
                2. Clean dust from fans
                3. Check for loose components
                4. Monitor CPU and GPU temperatures
                5. Replace failing fans if necessary
                6. Back up data if hard drive noise
                7. Consider professional repair""",
            
            "How do I backup my data?": """
                Recommended backup methods:
                1. Use Windows Backup
                2. Cloud storage (OneDrive, Google Drive)
                3. External hard drive
                4. Regular automated backups
                5. Keep multiple backup copies
                6. Test backup restoration
                7. Use versioning for important files""",
            
            "How do I free up disk space?": """
                Steps to free up disk space:
                1. Use Disk Cleanup utility
                2. Uninstall unnecessary programs
                3. Clear temporary files
                4. Empty Recycle Bin
                5. Remove old Windows updates
                6. Use Storage Sense
                7. Move files to external storage
                8. Compress large files""",
            
            "How do I enable two-factor authentication?": """
                Setting up 2FA:
                1. Access account security settings
                2. Choose authentication method
                3. Install authenticator app if needed
                4. Scan QR code or enter setup key
                5. Save backup codes
                6. Test the setup
                7. Enable on all important accounts""",
            
            "How do I secure my webcam?": """
                Webcam security measures:
                1. Use physical camera cover
                2. Check app permissions
                3. Keep drivers updated
                4. Use antivirus software
                5. Enable firewall
                6. Monitor active applications
                7. Consider dedicated security software""",
            
            "How do I uninstall programs?": """
                To uninstall programs:
                1. Open Control Panel
                2. Go to Programs and Features
                3. Select program to remove
                4. Click Uninstall
                5. Follow prompts
                6. Clean registry if needed
                7. Restart computer
                Alternative: Use Windows Settings > Apps""",
            
            "Program won't install properly": """
                Troubleshoot installation issues:
                1. Run as administrator
                2. Check system requirements
                3. Disable antivirus temporarily
                4. Clear temporary files
                5. Use compatibility mode
                6. Check disk space
                7. Update Windows
                8. Download from official source""",
            
            "Windows updates keep failing": """
                Fix Windows Update issues:
                1. Run Windows Update Troubleshooter
                2. Clear Windows Update cache
                3. Check disk space
                4. Disable antivirus temporarily
                5. Run DISM and SFC scans
                6. Reset Windows Update components
                7. Install updates manually if needed""",
            
            "How do I recover deleted files?": """
                File recovery options:
                1. Check Recycle Bin
                2. Use File History if enabled
                3. Check OneDrive/cloud backups
                4. Use Windows Previous Versions
                5. Try recovery software
                6. Restore from backup
                7. Contact data recovery service
                Note: Act quickly for better chances""",
        }

    def _generate_tech_follow_ups(self):  
        follow_ups = {}
        
        for answer in self.answers:
            if any(term in answer.lower() for term in ['internet', 'wifi', 'network']):
                follow_ups[answer] = [
                    "Have you tried resetting your router?",
                    "What speed does your internet plan provide?",
                    "Are other devices experiencing the same issue?"
                ]
            elif any(term in answer.lower() for term in ['software', 'program', 'update']):
                follow_ups[answer] = [
                    "When was the last system update?",
                    "How much free disk space do you have?",
                    "Are you running any antivirus software?"
                ]
            elif any(term in answer.lower() for term in ['printer', 'hardware', 'device']):
                follow_ups[answer] = [
                    "When did you last update the drivers?",
                    "Is the device recognized in Device Manager?",
                    "Have you checked all physical connections?"
                ]
        
        return follow_ups

    def preprocess_text(self, text):   
        words = text.lower().split()
        processed_words = []
        for word in words:
            if word in self.tech_terms:
                processed_words.append(self.tech_terms[word])
            else:
                processed_words.append(word)
        text = ' '.join(processed_words)
        
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]
        
        return ' '.join(tokens)

    def analyze_sentiment(self, text):     
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity <= -0.5:
            return 'very_negative'
        elif polarity < 0:
            return 'negative'
        elif polarity <= 0.3:
            return 'neutral'
        else:
            return 'positive'

    def get_response(self, user_question, threshold=0.3, top_n=3):    
        sentiment = self.analyze_sentiment(user_question)
        response_template = np.random.choice(self.sentiment_responses[sentiment])
        
        processed_question = self.preprocess_text(user_question)
        
        question_vector = self.vectorizer.transform([processed_question])
        
        similarities = cosine_similarity(question_vector, self.question_vectors)[0]
        
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        responses = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                answer = self.answers[idx]
                formatted_answer = f"{response_template}\n\n{answer}"
                follow_ups = self.follow_ups.get(answer, [])
                responses.append((formatted_answer, score, follow_ups))
        
        return responses if responses else []

    def chat(self):  
        print("\nTech Bot: Hello! I'm here to help with your technical issues. "
              "(type 'quit' to exit, 'history' to see conversation history)")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Tech Bot: Thanks for using our support service. Have a great day!")
                break
            
            elif user_input.lower() == 'history':
                self._display_history()
                continue
            
            responses = self.get_response(user_input)
            
            if responses:
                best_response, confidence, follow_ups = responses[0]
                print(f"\nTech Bot: {best_response}")
                
                sentiment = self.analyze_sentiment(user_input)
                self.log_conversation(user_input, best_response, confidence, sentiment)
                
                if confidence < 0.5:
                    print("\n(Note: This might not fully address your issue. Consider these alternatives:)")
                    for response, conf, _ in responses[1:]:
                        print(f"- {response}")
                
                if follow_ups:
                    print("\nAdditional troubleshooting questions:")
                    for q in follow_ups:
                        print(f"- {q}")
            else:
                response = ("I apologize, but I don't have specific information about that issue. "
                          "Would you like me to connect you with a human technician?")
                print(f"\nTech Bot: {response}")
                self.log_conversation(user_input, response, 0.0, 'neutral')

    def log_conversation(self, user_input, bot_response, confidence, sentiment):  
        self.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'confidence': confidence,
            'sentiment': sentiment
        })

    def _display_history(self):
        print("\nConversation History:")
        for entry in self.conversation_history:
            print(f"\nTime: {entry['timestamp']}")
            print(f"You: {entry['user_input']}")
            print(f"Bot: {entry['bot_response']}")
            print(f"Confidence: {entry['confidence']:.2f}")
            print(f"Sentiment: {entry['sentiment']}")

chatbot = TechSupportChatbot()

chatbot.chat()