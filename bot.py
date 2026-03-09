import io
import os
import wave
import asyncio
import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient, AudioSink
from dotenv import load_dotenv
import whisper

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
model = whisper.load_model("base")


class PerUserWaveSink(AudioSink):
    """Records each user's audio into a separate in-memory WAV buffer."""

    CHANNELS = 2
    SAMPLE_WIDTH = 2
    SAMPLING_RATE = 48000

    def __init__(self):
        super().__init__()
        self._buffers = {}   # user_id -> BytesIO
        self._writers = {}   # user_id -> wave.Wave_write
        self._decoders = {}  # user_id -> discord.opus.Decoder

    def wants_opus(self):
        # Return True so the library skips its own decoder.
        # We decode per-user ourselves and can catch corrupted packets.
        return True

    def write(self, user, data):
        if user is None:
            return
        opus_bytes = data.opus
        if not opus_bytes:
            return
        user_id = user.id
        if user_id not in self._decoders:
            buf = io.BytesIO()
            self._buffers[user_id] = buf
            wf = wave.open(buf, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.SAMPLING_RATE)
            self._writers[user_id] = wf
            self._decoders[user_id] = discord.opus.Decoder()
        try:
            pcm = self._decoders[user_id].decode(opus_bytes, fec=False)
            self._writers[user_id].writeframes(pcm)
        except Exception:
            pass  # skip corrupted or undecodable packets without stopping

    def cleanup(self):
        for wf in self._writers.values():
            try:
                wf.close()
            except Exception:
                pass

    @property
    def audio_data(self):
        """Returns {user_id: BytesIO} with complete WAV data per user."""
        result = {}
        for user_id, buf in self._buffers.items():
            buf.seek(0)
            result[user_id] = buf
        return result


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_voice_state_update(member, before, after):
    if member == bot.user:
        print(f"[VOICE] {before.channel} → {after.channel}")


@bot.command()
async def hello(ctx):
    await ctx.send(f"hello cousin {ctx.author.display_name}!")


@bot.command()
async def join(ctx):
    if not ctx.author.voice:
        return await ctx.send("You need to be in a voice channel first!")

    if ctx.voice_client:
        if ctx.voice_client.is_connected():
            await ctx.voice_client.move_to(ctx.author.voice.channel)
            return await ctx.send(f"Moved to **{ctx.author.voice.channel.name}**")
        else:
            await ctx.voice_client.disconnect(force=True)
            await asyncio.sleep(1)

    await asyncio.sleep(0.5)
    for attempt in range(2):
        try:
            vc = await ctx.author.voice.channel.connect(reconnect=False, cls=VoiceRecvClient)
            break
        except Exception as e:
            if attempt == 0 and "4006" in str(e):
                await asyncio.sleep(2)
                continue
            return await ctx.send(f"Failed to connect: `{e}`")
    await ctx.send(f"Joined **{vc.channel.name}**")


@bot.command()
async def leave(ctx):
    if not ctx.voice_client:
        return await ctx.send("I'm not in a voice channel!")
    await ctx.voice_client.disconnect()
    await ctx.send("Left the voice channel.")


@bot.command()
async def listen(ctx):
    if not ctx.voice_client or not ctx.voice_client.is_connected():
        return await ctx.send("I'm not in a voice channel! Use `!join` first.")
    if ctx.voice_client.is_listening():
        return await ctx.send("Already listening!")

    sink = PerUserWaveSink()

    def after(error):
        if error:
            print(f"Recording error: {error}")
        asyncio.run_coroutine_threadsafe(finished_callback(sink, ctx), bot.loop)

    ctx.voice_client.listen(sink, after=after)
    await ctx.send("Listening... use `!stop` to transcribe.")


@bot.command()
async def stop(ctx):
    if not ctx.voice_client or not ctx.voice_client.is_listening():
        return await ctx.send("Not currently listening.")
    ctx.voice_client.stop_listening()
    await ctx.send("Processing audio...")


async def finished_callback(sink: PerUserWaveSink, ctx: commands.Context):
    for user_id, audio_buf in sink.audio_data.items():
        filename = f"recording_{user_id}.wav"
        with open(filename, "wb") as f:
            f.write(audio_buf.read())

        result = await bot.loop.run_in_executor(
            None, lambda: model.transcribe(filename)
        )
        os.remove(filename)

        user = await bot.fetch_user(user_id)
        text = result["text"].strip() or "*(no speech detected)*"
        await ctx.send(f"**{user.display_name}**: {text}")


bot.run(os.environ["DISCORD_TOKEN"])
