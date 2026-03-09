import 'dotenv/config';
import { Client, GatewayIntentBits, Message, TextChannel } from 'discord.js';
import {
    joinVoiceChannel,
    VoiceConnectionStatus,
    entersState,
    getVoiceConnection,
    EndBehaviorType,
    VoiceConnection,
} from '@discordjs/voice';
import { writeFileSync, unlinkSync, existsSync, readFileSync, mkdirSync } from 'fs';
import path from 'path';

// prism-media has no TypeScript types
// eslint-disable-next-line @typescript-eslint/no-require-imports
const prism = require('prism-media');


const CHUNK_MS           = 1 * 60 * 1_000;          // 5 minutes
const TRANSCRIPT_CHANNEL = 'session-transcripts';
const WIKI_CHANNEL       = 'wiki';
const VOCAB_FILE         = 'vocabulary.json';
const OLLAMA_URL         = process.env.OLLAMA_URL    ?? 'http://localhost:11434';
const OLLAMA_MODEL       = process.env.OLLAMA_MODEL  ?? 'llama3.2';
const RECORDING_DIR      = process.env.RECORDING_DIR ?? 'E:/dnd_recordings';

// Ensure recording directory exists on startup
mkdirSync(RECORDING_DIR, { recursive: true });

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.GuildVoiceStates,
    ],
});

// Proper nouns and domain terms extracted from #wiki at startup
const vocabulary = new Set<string>();

interface GuildRecording {
    buffers: Map<string, Buffer[]>;
    subscriptions: Set<string>;
    speakingListener: (userId: string) => void;
    intervalId: NodeJS.Timeout | null;
    chunkNumber: number;
    flushing: boolean;
    hostUserId: string;   // user who called !listen — their stream gets diarization labels
    diarize: boolean;     // false when !listen nodiarize — skips diarization for everyone
}

const recordings = new Map<string, GuildRecording>();

// ── Events ────────────────────────────────────────────────────────────────────

client.once('ready', () => {
    console.log(`Logged in as ${client.user!.tag}`);
    buildVocabulary().catch(e => console.error('[vocab] Build failed:', e));
});

client.on('voiceStateUpdate', (oldState, newState) => {
    if (oldState.member?.id === client.user!.id || newState.member?.id === client.user!.id) {
        console.log(`[VOICE] ${oldState.channel?.name ?? 'null'} → ${newState.channel?.name ?? 'null'}`);
    }
});

client.on('messageCreate', async (msg: Message) => {
    if (msg.author.bot || !msg.content.startsWith('!') || !msg.guildId) return;
    const [cmd] = msg.content.slice(1).trim().toLowerCase().split(/ +/);
    try {
        if      (cmd === 'hello')  await msg.reply(`hello cousin ${msg.member?.displayName ?? msg.author.username}!`);
        else if (cmd === 'join')   await handleJoin(msg);
        else if (cmd === 'leave')  await handleLeave(msg);
        else if (cmd === 'listen') await handleListen(msg);
        else if (cmd === 'stop')   await handleStop(msg);
    } catch (e) {
        console.error(e);
        await msg.reply(`Error: \`${e}\``).catch(() => {});
    }
});

// ── Commands ──────────────────────────────────────────────────────────────────

async function handleJoin(msg: Message) {
    const channel = msg.member?.voice.channel;
    if (!channel) return msg.reply('You need to be in a voice channel first!');

    const guildId = msg.guildId!;
    const existing = getVoiceConnection(guildId);

    // If there's a stale (non-ready) connection, destroy it before reconnecting
    if (existing && existing.state.status !== VoiceConnectionStatus.Ready) {
        existing.destroy();
        await delay(1000);
    }

    await delay(500);

    const connection = joinVoiceChannel({
        channelId: channel.id,
        guildId,
        adapterCreator: msg.guild!.voiceAdapterCreator,
        selfDeaf: false, // must be false to receive audio
    });

    try {
        await entersState(connection, VoiceConnectionStatus.Ready, 10_000);
        await msg.reply(`Joined **${channel.name}**`);
    } catch (e) {
        connection.destroy();
        await msg.reply(`Failed to connect: \`${e}\``);
    }
}

async function handleLeave(msg: Message) {
    const connection = getVoiceConnection(msg.guildId!);
    if (!connection) return msg.reply("I'm not in a voice channel!");
    cleanupRecording(msg.guildId!, connection);
    connection.destroy();
    await msg.reply('Left the voice channel.');
}

async function handleListen(msg: Message) {
    const guildId = msg.guildId!;
    const connection = getVoiceConnection(guildId);

    if (!connection || connection.state.status !== VoiceConnectionStatus.Ready) {
        return msg.reply("I'm not in a voice channel! Use `!join` first.");
    }
    if (recordings.has(guildId)) {
        return msg.reply('Already listening!');
    }

    const args = msg.content.slice(1).trim().toLowerCase().split(/ +/);
    const diarize = !args.includes('nodiarize');
    const secsArg = args.slice(1).find(a => /^\d+$/.test(a));
    const chunkMs = secsArg ? parseInt(secsArg) * 1000 : CHUNK_MS;

    const recording: GuildRecording = {
        buffers: new Map(),
        subscriptions: new Set(),
        speakingListener: () => {},
        intervalId: null,
        chunkNumber: 0,
        flushing: false,
        hostUserId: msg.author.id,
        diarize,
    };
    recordings.set(guildId, recording);

    const receiver = connection.receiver;

    recording.speakingListener = (userId: string) => {
        if (!recordings.has(guildId)) return;
        if (recording.subscriptions.has(userId)) return;

        recording.subscriptions.add(userId);
        recording.buffers.set(userId, []);

        const opusStream = receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.Manual },
        });

        const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });

        // Swallow decode errors (RTCP packets, corrupted frames) so a bad packet
        // doesn't propagate as an unhandled stream error and crash the process.
        opusStream.on('error', (e: Error) => console.warn(`[opus] Stream error (${userId}): ${e.message}`));
        decoder.on('error',    (e: Error) => console.warn(`[opus] Decode error (${userId}): ${e.message}`));

        opusStream.pipe(decoder).on('data', (chunk: Buffer) => {
            recording.buffers.get(userId)?.push(chunk);
        });

        opusStream.once('close', () => recording.subscriptions.delete(userId));
    };

    receiver.speaking.on('start', recording.speakingListener);

    // Pre-subscribe to everyone already in the channel (speaking event is unreliable)
    const voiceChannel = msg.member?.voice.channel;
    if (voiceChannel) {
        for (const [memberId] of voiceChannel.members) {
            if (memberId === client.user!.id) continue;
            recording.speakingListener(memberId);
        }
    }

    recording.intervalId = setInterval(() => {
        flushChunk(guildId).catch(e => console.error('[chunk flush error]', e));
    }, chunkMs);

    const mode = diarize ? 'with speaker diarization' : 'without diarization';
    const chunkSecs = chunkMs / 1000;
    await msg.reply(`Listening (${mode}, chunks every ${chunkSecs}s)... transcripts sent to **#${TRANSCRIPT_CHANNEL}**. Use \`!stop\` to flush and stop.`);
}

async function handleStop(msg: Message) {
    const guildId = msg.guildId!;
    const recording = recordings.get(guildId);
    if (!recording) return msg.reply('Not currently listening.');

    const connection = getVoiceConnection(guildId);
    cleanupRecording(guildId, connection ?? undefined);

    // Wait for any in-progress interval flush to finish before doing the final one
    for (let i = 0; i < 60 && recording.flushing; i++) {
        await delay(500);
    }

    await msg.reply('Processing final audio chunk...');
    await flushChunk(guildId, recording);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function flushChunk(guildId: string, detached?: GuildRecording) {
    const recording = detached ?? recordings.get(guildId);
    if (!recording || recording.flushing) return;
    recording.flushing = true;

    const chunkNum = ++recording.chunkNumber;

    // Snapshot buffers and immediately reset so audio keeps flowing during transcription
    const snapshot = new Map<string, Buffer[]>();
    for (const [userId, chunks] of recording.buffers) {
        snapshot.set(userId, chunks);
        recording.buffers.set(userId, []);
    }

    try {
        const guild = client.guilds.cache.get(guildId);
        const rawChannel = guild?.channels.cache.find(c => c.name === TRANSCRIPT_CHANNEL);

        if (!rawChannel?.isSendable()) {
            console.warn(`[flush] #${TRANSCRIPT_CHANNEL} not found or not sendable in guild ${guildId}`);
            return;
        }

        const parts: string[] = [];

        for (const [userId, chunks] of snapshot) {
            if (chunks.length === 0) continue;

            const filename = path.join(RECORDING_DIR, `recording_${userId}_chunk${chunkNum}.wav`);
            writeFileSync(filename, makeWav(chunks));

            try {
                const result = await transcribe(filename, recording.diarize && userId === recording.hostUserId);
                unlinkSync(filename);
                if (result.text) {
                    const user = await client.users.fetch(userId);
                    if (result.diarized && userId === recording.hostUserId) {
                        // Conference mic host — keep SPEAKER labels
                        parts.push(`**${user.displayName}** (room):\n${result.text}`);
                    } else {
                        // Remote user — always use their Discord name, strip any stray labels
                        const textOnly = result.diarized
                            ? result.text.replace(/^\[SPEAKER_\d+\]: /gm, '').trim()
                            : result.text;
                        parts.push(`**${user.displayName}**: ${textOnly}`);
                    }
                }
            } catch (e) {
                console.error(`Transcription error for ${userId} chunk ${chunkNum}:`, e);
                try { unlinkSync(filename); } catch { /* ignore */ }
            }
        }

        if (parts.length > 0) {
            await sendLong(rawChannel, `**— Chunk ${chunkNum} —**\n${parts.join('\n')}`);
        }
    } finally {
        recording.flushing = false;
    }
}

function cleanupRecording(guildId: string, connection?: VoiceConnection) {
    const recording = recordings.get(guildId);
    if (!recording) return;
    recordings.delete(guildId);

    if (recording.intervalId) {
        clearInterval(recording.intervalId);
        recording.intervalId = null;
    }

    if (connection) {
        connection.receiver.speaking.off('start', recording.speakingListener);
        for (const userId of recording.subscriptions) {
            connection.receiver.subscriptions.get(userId)?.destroy();
        }
    }
}

// ── Vocabulary ────────────────────────────────────────────────────────────────

async function buildVocabulary() {
    // Load cached vocab so the bot is immediately useful even before re-scanning
    if (existsSync(VOCAB_FILE)) {
        const cached = JSON.parse(readFileSync(VOCAB_FILE, 'utf8')) as string[];
        for (const w of cached) vocabulary.add(w);
        console.log(`[vocab] Loaded ${vocabulary.size} cached terms from ${VOCAB_FILE}`);
    }

    for (const guild of client.guilds.cache.values()) {
        // Ensure the channel list is populated (may be stale on startup)
        await guild.channels.fetch();

        const wikiChannel = guild.channels.cache.find(c => c.name === WIKI_CHANNEL);
        if (!wikiChannel || !(wikiChannel instanceof TextChannel)) {
            console.log(`[vocab] No #${WIKI_CHANNEL} text channel in "${guild.name}", skipping`);
            continue;
        }

        console.log(`[vocab] Scanning #${WIKI_CHANNEL} in "${guild.name}"...`);
        const messages = await fetchAllMessages(wikiChannel);
        console.log(`[vocab] ${messages.length} messages found — extracting terms via Ollama (${OLLAMA_MODEL})...`);

        const texts = messages.map(m => m.content).filter(Boolean);
        const batches = batchText(texts, 2000);

        for (let i = 0; i < batches.length; i++) {
            console.log(`[vocab] Batch ${i + 1}/${batches.length}...`);
            const terms = await extractTermsViaOllama(batches[i]);
            for (const t of terms) vocabulary.add(t);
        }
    }

    writeFileSync(VOCAB_FILE, JSON.stringify([...vocabulary], null, 2));
    console.log(`[vocab] Done — ${vocabulary.size} unique terms saved to ${VOCAB_FILE}`);
}

async function fetchAllMessages(channel: TextChannel): Promise<Message[]> {
    const all: Message[] = [];
    let before: string | undefined;

    for (;;) {
        const batch = await channel.messages.fetch({ limit: 100, ...(before ? { before } : {}) });
        if (batch.size === 0) break;
        all.push(...batch.values());
        const last = batch.last();
        if (!last || batch.size < 100) break;
        before = last.id;
    }

    return all;
}

async function extractTermsViaOllama(text: string): Promise<string[]> {
    const prompt =
        `You are building a speech-recognition vocabulary list from D&D campaign notes.\n` +
        `Extract ONLY words that a speech-to-text model would likely misspell or mishear ` +
        `because they are invented, fictional, or highly unusual:\n` +
        `  - Proper names: characters, NPCs, deities, places, planes, organisations\n` +
        `  - Invented/fantasy words: made-up spells, items, creatures, languages\n` +
        `  - Obscure proper D&D terminology that would not appear in a standard dictionary (e.g. "Bigby's Hand", "Mystra", "Vecna")\n\n` +
        `Do NOT include:\n` +
        `  - Common English words even if used in a fantasy context (fire, damage, pain, death, shadow, curse, storm, ancient, divine, arcane, infernal, burning, stress, extreme, etc.)\n` +
        `  - Generic D&D terms that appear in any dictionary (wizard, dragon, spell, sword, dungeon, tavern, cleric, paladin, etc.)\n` +
        `  - Adjectives, verbs, or descriptive phrases\n\n` +
        `Return ONLY a comma-separated list of the extracted words. No explanation, no numbering, nothing else.\n\n` +
        `Text:\n${text}`;

    try {
        const res = await fetch(`${OLLAMA_URL}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: OLLAMA_MODEL, prompt, stream: false }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json() as { response: string };

        return data.response
            .split(',')
            .map((w: string) => w.trim().replace(/^["'\s]+|["'\s]+$/g, ''))
            .filter((w: string) => w.length > 1);

    } catch (e) {
        console.error('[vocab] Ollama extraction error:', e);
        return [];
    }
}

// Split an array of message strings into batches no larger than maxChars
function batchText(texts: string[], maxChars: number): string[] {
    const batches: string[] = [];
    let current = '';

    for (const text of texts) {
        if (current.length + text.length > maxChars && current) {
            batches.push(current);
            current = '';
        }
        current += (current ? '\n' : '') + text;
    }

    if (current) batches.push(current);
    return batches;
}

// ── Audio / Transcription ─────────────────────────────────────────────────────

function makeWav(chunks: Buffer[]): Buffer {
    const pcm = Buffer.concat(chunks);
    const hdr = Buffer.alloc(44);
    hdr.write('RIFF', 0);
    hdr.writeUInt32LE(pcm.length + 36, 4);
    hdr.write('WAVE', 8);
    hdr.write('fmt ', 12);
    hdr.writeUInt32LE(16, 16);       // fmt chunk size
    hdr.writeUInt16LE(1, 20);        // PCM format
    hdr.writeUInt16LE(2, 22);        // 2 channels (stereo)
    hdr.writeUInt32LE(48000, 24);    // sample rate
    hdr.writeUInt32LE(192000, 28);   // byte rate = 48000 * 2 * 2
    hdr.writeUInt16LE(4, 32);        // block align = 2 channels * 2 bytes
    hdr.writeUInt16LE(16, 34);       // bits per sample
    hdr.write('data', 36);
    hdr.writeUInt32LE(pcm.length, 40);
    return Buffer.concat([hdr, pcm]);
}

interface TranscribeResult {
    text: string;
    diarized: boolean;
}

async function transcribe(filename: string, diarize = false): Promise<TranscribeResult> {
    const port = process.env.TRANSCRIBE_PORT ?? '8765';
    const res = await fetch(`http://127.0.0.1:${port}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, diarize }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText })) as { error: string };
        throw new Error(`Transcribe server: ${err.error}`);
    }
    return await res.json() as TranscribeResult;
}

async function sendLong(channel: { send: (s: string) => Promise<unknown> }, text: string) {
    const MAX = 2000;
    if (text.length <= MAX) { await channel.send(text); return; }

    const chunks: string[] = [];
    let current = '';

    for (const line of text.split('\n')) {
        // Hard-split any single line that exceeds MAX on its own
        if (line.length > MAX) {
            if (current) { chunks.push(current); current = ''; }
            for (let i = 0; i < line.length; i += MAX) chunks.push(line.slice(i, i + MAX));
            continue;
        }
        const next = current ? current + '\n' + line : line;
        if (next.length > MAX) {
            chunks.push(current);
            current = line;
        } else {
            current = next;
        }
    }
    if (current) chunks.push(current);

    for (const chunk of chunks) await channel.send(chunk);
}

function delay(ms: number) {
    return new Promise<void>(resolve => setTimeout(resolve, ms));
}

// ── Start ─────────────────────────────────────────────────────────────────────

client.login(process.env.DISCORD_TOKEN!);
