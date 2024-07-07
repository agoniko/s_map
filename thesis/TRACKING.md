# Progress tracking of the thesis

- Type: Master Thesis
- Student: Family name, Name
- [https://gitlab-rbl.unipv.it/robolab/thesis-templatex](Repository on Gitlab)

# Status

Number | Chapter        | Version | Status  | Responsibility
------ | -------------- | ------- | ------- | --------------
0      | Abstract       | v1      | WRITING | Student
1      | Intro          | v1      | WRITING | Student
2      | SOTA           | v1      | WRITING | Student
3      | Tools          | v1      | WRITING | Student
4      | Implementation | v1      | WRITING | Student
5      | Results        | v1      | WRITING | Student
6      | Conclusions    | v1      | WRITING | Student

# Detailed structure

Abstract
- Title: Introduction
- File: `abstract.inc.tex`
- Content: Abstract (see the template for guidelines).

Chapter 1
- Title: Introduction
- File: `chapter_intro.inc.tex`
- Content: Context; goals; organization of the document.

Chapter 2
- Title: State of the art
- File: `chapter_sota.inc.tex`
- Content: References to existing works and papers.

Chapter 3
- Title: Tools and Frameworks
- File: `chapter_tools.inc.tex`
- Content: List of tools and frameworks used in the thesis.

Chapter 4
- Title: Implementation
- File: `chapter_imlementation.inc.tex`
- Content: Description of the implementation; code and its organization.

Chapter 5
- Title: Results
- File: `chapter_results.inc.tex`
- Content: Tables and graphs showing the results.

Chapter 6
- Title: Conclusions
- File: `chapter_conclusions.inc.tex`
- Content: Conclusions and future works.

# Using this document

## The `status` field

The `status` field can take one of the following values:

- WRITING: Initial writing contents are required to be completed by the student.
- REVIEW: Contents are complete; nothing more to add by the student; chapter ready to be revised (or under revision).
- REVISED: Review done by the Professor; there are observations and comments to address.
- UPDATE: The student is addressing the comments.
- FINAL: The chapter completed; no more changes are required.
	
## How to track the progress
      
Every change of the status must be notified by email or other messaging tool by the responsible.

1. Until the content is completed, the chapter stays in the WRITING state; responsibility to Student.
2. Once the content is completed, the status moves to REVIEW; responsibility to Professor. IMPORTANT: in REVIEW state the Student **MUST NOT EDIT** the chapter until he/she receives a notification from the Professor.
3. Once the revision is done, the Professor changes the status to REVISED and notifies the Student; the Student is required to address the observations/comments; responsibility is of the Student.
4. When the Student starts the update, he/she moves the status to UPDATE and increases the Version number.
5. Once the update is done by the Student, the thesis returns is set to REVIEW state by the Student, so that the Professor can check the updates; responsibility to Professor.
6. If the chapter does not need any further change, the Professor moves the chapter in the FINAL state; otherwise the state becomes REVISED and the process restart from (3).

During the update stage, each comment must be properly addressed and a short note shall be reported within the Tex file or in an email.
