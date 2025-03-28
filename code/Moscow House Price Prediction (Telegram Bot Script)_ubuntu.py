import asyncio
import numpy as np
import cv2
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext
import joblib
import nest_asyncio
import os

MOSCOW_HOUSE_PRICE_MODEL_PATH = "/root/moscow_house_price/Moscow-House-Price-Prediction/models/moscow_housing_price_hgb.joblib"
moscow_house_price_model = joblib.load(MOSCOW_HOUSE_PRICE_MODEL_PATH)

# Telegram bot token
TOKEN = "7381999469:AAHjRTQ7fHyZ8xOqUFBykv_Hr5OlNvP_KwU"
app = Application.builder().token(TOKEN).build()

# Order of inputs required for prediction
feature_order = [
    'minutes_to_metro', 'number_of_rooms', 'area', 'living_area',
    'kitchen_area', 'floor', 'number_of_floors', 'apartment_type_code',
    'region_code', 'renovation_code'
]

# Mapping categorical options to numerical values
categorical_mappings = {
    'apartment_type_code': {"Secondary": 1, "New building": 0},
    'region_code': {"Moscow region": 1, "Moscow": 0},
    'renovation_code': {"Cosmetic": 0, "European-style renovation": 2, "Without renovation": 3, "Designer": 1}
}

# Standard deviation constants for price range calculation
price_std = {"Moscow": 10658837, "Moscow region": 2676070}

# Store user responses
user_data = {}

def round_to_thousand(n):
    return int(np.round(n / 1000) * 1000)

# START command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "\U0001F3E1 Welcome to the Moscow House Price Estimator! \U0001F3E1\n\n"
        "This bot helps you estimate house prices in Moscow.\n"
        "To start, use /gethouseprice and follow the prompts."
    )

# HELP command
async def help_command(update: Update, context: CallbackContext):
    await start(update, context)

# GET HOUSE PRICE command
async def get_house_price(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    user_data[chat_id] = {}
    await update.message.reply_text("\U0001F687 How many minutes to the nearest metro station? (Example: 10)")

# Handle user inputs
async def collect_features(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    feature_index = len(user_data[chat_id])
    
    if feature_index >= len(feature_order):
        return
    
    current_feature = feature_order[feature_index]
    
    if current_feature in categorical_mappings:
        await ask_categorical(update, chat_id, current_feature)
    else:
        await ask_numeric(update, chat_id, current_feature)

# Ask for categorical features
async def ask_categorical(update: Update, chat_id, feature):
    options = categorical_mappings[feature]
    keyboard = [[InlineKeyboardButton(k, callback_data=k)] for k in options]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"\U0001F4CB Choose {feature.replace('_', ' ')}:", reply_markup=reply_markup)

# Handle button clicks for categorical choices
async def handle_button(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    feature_index = len(user_data[chat_id])
    feature = feature_order[feature_index]
    
    user_data[chat_id][feature] = categorical_mappings[feature][query.data]
    await ask_next_feature(query, chat_id)

# Ask for numeric features
async def ask_numeric(update: Update, chat_id, feature):
    await update.message.reply_text(f"\U0001F4CF Enter {feature.replace('_', ' ')}:")

# Process user response for numeric inputs
async def process_numeric_input(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    feature_index = len(user_data[chat_id])
    feature = feature_order[feature_index]
    
    try:
        value = float(update.message.text) if '.' in update.message.text else int(update.message.text)
        user_data[chat_id][feature] = value
        await ask_next_feature(update, chat_id)
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a valid number!")

# Move to the next question or predict price
async def ask_next_feature(update, chat_id):
    feature_index = len(user_data[chat_id])

    if feature_index < len(feature_order):
        next_feature = feature_order[feature_index]

        if next_feature in categorical_mappings:
            await ask_categorical(update, chat_id, next_feature)
        else:
            await update.message.reply_text(f"📊 Enter {next_feature.replace('_', ' ')}:")
    else:
        await predict_price(update, chat_id)

# Predict house price
async def predict_price(update, chat_id):
    features = [user_data[chat_id][col] for col in feature_order]
    region = "Moscow" if user_data[chat_id]["region_code"] == 0 else "Moscow region"
    
    predicted_price = moscow_house_price_model.predict([features])[0]
    
    std_dev = price_std[region]
    price_range_min = predicted_price - (std_dev / (user_data[chat_id]["area"] * 0.5))
    price_range_max = predicted_price + (std_dev / (user_data[chat_id]["area"] * 0.5))
    
    price_range_min = round_to_thousand(price_range_min)
    price_range_max = round_to_thousand(price_range_max)
    predicted_price = round_to_thousand(predicted_price)
    
    await update.message.reply_text(
        f"💰 Estimated Price: {predicted_price:,} RUB\n"
        f"📉 Possible Range: {price_range_min:,} - {price_range_max:,} RUB\n"
        f"🏙️ Region: {region}\n"
    )
    del user_data[chat_id]

# Register handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("gethouseprice", get_house_price))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_numeric_input))
app.add_handler(CallbackQueryHandler(handle_button))

# Run bot
if __name__ == "__main__":
    nest_asyncio.apply()
    app.run_polling()
