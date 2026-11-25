from pathlib import Path

BASE_DIR = Path("/data/brhanu/thesis_project")
tex_file = BASE_DIR / "thesis_v3_updated.tex"

def main():
    content = tex_file.read_text(encoding='utf-8')
    
    # 1. Update the section title
    content = content.replace(
        "\\section{Human-Verified Subset Evaluation (E1--E4)}",
        "\\section{Human-Verified Subset Evaluation (E1--E9)}"
    )
    
    # 2. Extract Experiment 9
    start_marker = "\\subsubsection{Experiment 9: Automated Second-Opinion Routing (Pilot Study)}"
    end_marker = "\\subsubsection{Completed Phase: YOLOv11-seg and SAM Integration}"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("❌ Error: Could not find Experiment 9 markers in the text.")
        return
        
    exp9_text = content[start_idx:end_idx].strip()
    
    # Remove Experiment 9 from its current position
    # (replace from start_marker to end_marker with just end_marker)
    content_removed = content[:start_idx] + content[end_idx:]
    
    # 3. Rename header to \subsection{E9: Automated Second-Opinion Routing (Pilot Study)}
    exp9_text_updated = exp9_text.replace(
        "\\subsubsection{Experiment 9: Automated Second-Opinion Routing (Pilot Study)}",
        "\\subsection{E9: Automated Second-Opinion Routing (Pilot Study)}"
    )
    
    # 4. Insert at the new position: right before Qualitative Archival Hermeneutics
    target_pos = "\\subsection{Qualitative Archival Hermeneutics: Collaborative Case Studies}"
    insert_idx = content_removed.find(target_pos)
    
    if insert_idx == -1:
        print("❌ Error: Could not find target position in the text.")
        return
        
    final_content = (
        content_removed[:insert_idx] + 
        exp9_text_updated + "\n\n" + 
        content_removed[insert_idx:]
    )
    
    tex_file.write_text(final_content, encoding='utf-8')
    print("✅ Successfully promoted Experiment 9 to Section 12 as E9!")

if __name__ == "__main__":
    main()
