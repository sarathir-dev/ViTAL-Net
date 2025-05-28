from utils.frame_extractor import extract_frames_from_videos

violence_input = "/datasets/hockey_fight/fight"
non_violence_input = "/datasets/hockey_fight/nonfight"

output_root = "/data/violence_dataset/hockey_fight"

extract_frames_from_videos(violence_input, output_root, "fight", frame_rate=3)
extract_frames_from_videos(
    non_violence_input, output_root, "nonfight", frame_rate=3)

print("Frane extraction complete.")
