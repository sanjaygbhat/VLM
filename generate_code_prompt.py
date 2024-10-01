import os
import argparse
import pyperclip

def generate_markdown(root_dir, extensions):
    """
    Traverses the root_dir and generates Markdown-formatted code blocks
    for files with specified extensions.

    Args:
        root_dir (str): The root directory to traverse.
        extensions (list): List of file extensions to include (e.g., ['.py', '.txt', '.sh']).

    Returns:
        str: The generated Markdown content.
    """
    markdown_content = []
    for subdir, dirs, files in os.walk(root_dir):
        # Compute relative path from root_dir
        rel_dir = os.path.relpath(subdir, root_dir)
        if rel_dir == ".":
            rel_dir = ""
        else:
            markdown_content.append(f"## `{rel_dir}` Directory\n\n")

        # Filter files based on the specified extensions
        target_files = [f for f in files if os.path.splitext(f)[1] in extensions]

        for file in target_files:
            file_path = os.path.join(rel_dir, file) if rel_dir else file
            markdown_content.append(f"### `{file}`\n\n")
            markdown_content.append(f"```{get_fence_language(file)}:{file_path}\n")
            try:
                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as f:
                    code = f.read()
                markdown_content.append(code)
            except Exception as e:
                markdown_content.append(f"<!-- Error reading file: {e} -->\n")
            markdown_content.append("```\n\n")

    return "\n".join(markdown_content)

def get_fence_language(file_name):
    """
    Determines the appropriate language identifier for Markdown code fences
    based on the file extension.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The language identifier for the code fence.
    """
    extension = os.path.splitext(file_name)[1].lower()
    if extension == '.py':
        return 'python'
    elif extension == '.sh':
        return 'bash'
    elif extension == '.txt':
        return 'text'
    else:
        return 'text'  # Default to text if unknown

def copy_to_clipboard(content):
    """
    Copies the given content to the system clipboard.

    Args:
        content (str): The content to copy.
    """
    try:
        pyperclip.copy(content)
        print("✅ Markdown content successfully copied to clipboard.")
    except pyperclip.PyperclipException as e:
        print(f"❌ Failed to copy to clipboard: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown prompt of your codebase.")
    parser.add_argument(
        'root_dir',
        type=str,
        nargs='?',
        default='.',
        help='Root directory of the codebase (default: current directory)'
    )
    parser.add_argument(
        '-e', '--extensions',
        type=str,
        nargs='+',
        default=['.py', '.txt', '.sh'],
        help='File extensions to include (e.g., .py .txt .sh)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output Markdown file path (default: none)'
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"❌ Error: The directory '{root_dir}' does not exist.")
        return

    extensions = args.extensions
    print(f"🔍 Generating Markdown for files with extensions: {', '.join(extensions)}\n")

    markdown = generate_markdown(root_dir, extensions)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"📝 Markdown prompt successfully written to '{args.output}'.")
            copy_to_clipboard(markdown)
        except Exception as e:
            print(f"❌ Error writing to file '{args.output}': {e}")
    else:
        print("📋 Copying Markdown content to clipboard...")
        copy_to_clipboard(markdown)
        print("\n📄 Markdown Content:\n")
        print(markdown)

if __name__ == "__main__":
    main()