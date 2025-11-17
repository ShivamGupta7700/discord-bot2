import discord 
from discord.ext import commands
import os
from dotenv import load_dotenv
# ===========================
# BOT CONFIG
# ===========================

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree


# ===========================
# ON READY → SYNC SLASH COMMANDS
# ===========================
@bot.event
async def on_ready():
    await tree.sync()
    print(f"🤖 Bot is online as {bot.user}")
    print("✔ Slash commands synced!")


# ===========================
# /upload_war — Upload Excel File
# ===========================
@tree.command(name="upload_war", description="Upload a war Excel (.xlsx) file")
async def upload_war(interaction: discord.Interaction, file: discord.Attachment):
    await interaction.response.defer()

    if not file.filename.endswith(".xlsx"):
        return await interaction.followup.send("❌ Please upload an **Excel .xlsx** file.")

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    await file.save(file_path)

    await interaction.followup.send(f"✅ War file uploaded and saved as **{file.filename}**!")


# ===========================
# /train_ai — Train ML model
# ===========================
@tree.command(name="train_ai", description="Train the AI model from all war data")
async def train_ai(interaction: discord.Interaction):
    await interaction.response.defer()

    try:
        import WarAITraning
        WarAITraning.train()
        await interaction.followup.send("🔥 AI model trained successfully and saved!")
    except Exception as e:
        await interaction.followup.send(f"❌ Training failed:\n```\n{e}\n```")


# ===========================
# /predict_ai — Predict next war stars
# ===========================
@tree.command(name="predict_ai", description="Predict next war performance for all players")
async def predict_ai(interaction: discord.Interaction):
    await interaction.response.defer()

    try:
        import Recomends
        msg = Recomends.run()   # returns prediction message text
        await interaction.followup.send(msg)
    except Exception as e:
        await interaction.followup.send(f"❌ Prediction failed:\n```\n{e}\n```")

@tree.command(name="top_attackers", description="Show the top predicted attackers for next war")
async def top_attackers(interaction: discord.Interaction):
    await interaction.response.defer()

    try:
        import Recomends
        final = Recomends.get_dataframe()

        # Sort by attack prediction
        final = final.sort_values(by="PREDICTED_STARS", ascending=False).head(10)

        embed = discord.Embed(
            title=" Top Attackers (Predicted Next War) ",
            color=discord.Color.orange()
        )

        for i, row in final.iterrows():
            embed.add_field(
                name=f"{row['NAME']} ({row['TAG']})",
                value=f"<:FullStar:1439860878693372016> **Predicted Stars:** {round(row['PREDICTED_STARS'],2)}\n"
                      f"💥 TH: {int(row['ATTACKER TH'])}",
                inline=False
            )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Error:\n```\n{e}\n```")

# ===========================
# RUN BOT
# ===========================

load_dotenv()

TOKEN = os.getenv("TOKEN")
bot.run(TOKEN)



