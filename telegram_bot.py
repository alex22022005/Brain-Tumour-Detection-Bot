import os
import io
import logging
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from ultralytics import YOLO

# --- Setup ---
# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# FIX: 'getLogger' must be called from the 'logging' module.
logger = logging.getLogger(__name__)

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_PATH = 'best.pt'

# IMPORTANT: Map your model's class indices (0, 1, 2) to readable names.
# Please verify these are the correct names for your trained model.
CLASS_NAMES = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'Pituitary Tumor'
    # Add more classes here if needed, e.g., 3: 'No Tumor'
}
# --- End Configuration ---

# --- Load Models ---
try:
    # Load the custom YOLOv8 model for tumor detection
    yolo_model = YOLO(MODEL_PATH)
    logger.info("Brain Tumor Detection model loaded successfully.")

    # Configure the Gemini API for the chatbot
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini model configured successfully.")

except Exception as e:
    logger.critical(f"A critical error occurred during model initialization: {e}")
    exit()

# --- Telegram Bot Handlers ---

def start(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message when the /start command is issued."""
    user_name = update.effective_user.first_name
    welcome_message = (f"👋 Hello, {user_name}!\n\n"
                       "I am the Brain Tumor Detection Bot. Please send me a brain scan image, and I will analyze it for potential tumor types.")
    update.message.reply_text(welcome_message)

def help_command(update: Update, context: CallbackContext) -> None:
    """Sends instructions when the /help command is issued."""
    help_text = ("**How to use me:**\n"
                 "1. **Analysis:** Send me a brain scan image (like an MRI).\n"
                 "2. **Follow-up:** After I provide the analysis, you can ask me questions about the detected tumor types for informational purposes.")
    update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

def handle_image(update: Update, context: CallbackContext) -> None:
    """Processes a brain scan image, runs detection, and sends back the annotated result."""
    chat_id = update.message.chat_id
    try:
        context.bot.send_message(chat_id, "Scan received. Analyzing... 🩺")

        # Download the image sent by the user
        photo_file = update.message.photo[-1].get_file()
        image_bytes = photo_file.download_as_bytearray()
        
        # Open the image with PIL
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Perform YOLOv8 inference
        results = yolo_model(pil_image)
        result = results[0]
        
        # Render the results on the image (draws bounding boxes and labels)
        annotated_image_np = result.plot()
        annotated_image_pil = Image.fromarray(annotated_image_np[..., ::-1]) # Convert BGR to RGB

        # Save the annotated image to a byte buffer to send back
        bio = io.BytesIO()
        bio.name = 'annotated_scan.jpeg'
        annotated_image_pil.save(bio, 'JPEG')
        bio.seek(0)
        
        detected_tumors = []
        # --- NEW FIX STARTS HERE ---
        # A more direct method to extract class information to prevent mismatches.
        if result.boxes is not None:
            class_indices = result.boxes.cls.tolist()  # Get all class IDs as a list
            for class_id in class_indices:
                tumor_name = CLASS_NAMES.get(int(class_id), "Unknown Type")
                detected_tumors.append(tumor_name)
        
        # Remove duplicates if any were found
        if detected_tumors:
            detected_tumors = sorted(list(set(detected_tumors)))
        # --- NEW FIX ENDS HERE ---
        
        # Store detected tumors in user context for the chatbot
        context.user_data['last_detected_tumors'] = detected_tumors
        
        if detected_tumors:
            caption_text = f"**Analysis Complete.**\n\nPotential findings: **{', '.join(detected_tumors)}**.\n\nYou can now ask me questions for more information."
        else:
            caption_text = "Analysis complete. I did not detect any of the tumor types I'm trained to recognize in this scan."
            context.user_data['last_detected_tumors'] = None

        update.message.reply_photo(photo=bio, caption=caption_text, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Processed scan for chat {chat_id}. Found: {', '.join(detected_tumors) if detected_tumors else 'None'}")

    except Exception as e:
        logger.error(f"Error processing image for chat {chat_id}: {e}", exc_info=True)
        update.message.reply_text("Sorry, I encountered an error during the analysis. Please try sending the scan again.")

def handle_text(update: Update, context: CallbackContext) -> None:
    """Handles text messages and uses Gemini for chatbot functionality about tumors."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    
    thinking_message = context.bot.send_message(chat_id=chat_id, text="Consulting knowledge base... 🧠")

    last_tumors = context.user_data.get('last_detected_tumors')
    
    # Construct a responsible prompt for the Gemini model
    if last_tumors:
        prompt = (f"You are a helpful medical information AI. A user's brain scan analysis has indicated a potential "
                  f"'{', '.join(last_tumors)}'. The user is now asking a follow-up question. "
                  f"Answer their question in a clear, simple, and reassuring tone. Provide general information only.\n\n"
                  f"User's Question: '{user_question}'")
    else:
        prompt = (f"You are a helpful medical information AI. A user is asking a general question about brain tumors. "
                  f"Answer their question in a clear, simple, and reassuring tone.\n\n"
                  f"User's Question: '{user_question}'")
    
    disclaimer = "\n\n**Important Disclaimer:** I am an AI assistant, not a medical professional. This information is for educational purposes only. Please consult a qualified doctor for diagnosis and medical advice."

    try:
        # Generate a response using the Gemini model
        response = gemini_model.generate_content(prompt)
        bot_answer = response.text + disclaimer
        
        context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text=bot_answer, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Answered text query for chat {chat_id}.")

    except Exception as e:
        logger.error(f"Error calling Gemini API for chat {chat_id}: {e}", exc_info=True)
        error_message = "I'm having trouble accessing my knowledge base right now. Please try again later." + disclaimer
        context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text=error_message, parse_mode=ParseMode.MARKDOWN)

def main() -> None:
    """Starts the Telegram bot."""
    if not TELEGRAM_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN not found in environment variables. Bot cannot start.")
        return

    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))
    # FIX: 'Filters' should be accessed directly, not via 'MessageHandler.Filters'
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    
    updater.start_polling()
    logger.info("Brain Tumor Detection Bot is polling...")
    updater.idle()

if __name__ == '__main__':
    main()

