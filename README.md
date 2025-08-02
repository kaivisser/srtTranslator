# srtTranslator

Simple program that translates English subtitles to Japanese (configured for Game of Thrones at the moment)
Usage:

Add llm provider private api key to .env file

```
nix-shell
cd translate
python3 main.py run Subtitles\ \(EN\)/S8/Game.of.Thrones.S08E01.720p.WEB.H264-MEMENTO-HI.srt 8 1
```

Will translate S8E1 subs to Japanese and place the folder in Subtitles(JP)/S8/Game of Thrones S8E1.jp.srt

